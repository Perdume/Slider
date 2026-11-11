"""Slider 2v2 규칙을 JAX 배열 연산으로 다시 쓴 것 (GPU/TPU/CPU 어디서나 돈다).

규칙은 rust/src/bitboard.rs 와 같다: 한 턴에 내 말 하나를 골라 타일 위를 걷고(다른 말이 있는 칸은 못 지남),
빈 칸 방향으로 서 있는 타일째 미끄러진다. 어느 말이든 골(3,3)에 멈추면 승리, 수가 없으면 통과,
같은 국면(차례 포함)이 3번 나오면 무승부. 수 제한은 없다 (안전장치 DEFAULT_MAX_PLIES 뿐).
무승부는 양쪽 다 진 것으로 센다 (`outcome_values`).

상태는 칸 49개짜리 bool 배열 3개 (tiles, me=둘 차례인 쪽 말, opp). 수 하나 = 평평한 인덱스
k*196 + from*4 + dir  (k = 둘 중 몇 번째 말인지, 칸 번호 오름차순). 이 순서는 Rust 의 generate 순서와 같다.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

N = 49
GOAL = 24
# 수 제한은 사실상 없앤 것과 같다: 골에 닿거나 같은 국면이 3번 나오면 끝나고, 이 값은 안전장치일 뿐이다.
# 이 값까지 가도 안 끝난 판은 무승부(둘 다 진 것)로 센다. 규칙 대조 때는 150 으로 줄여서 쓴다.
DEFAULT_MAX_PLIES = 4096
NUM_MOVES = 2 * N * 4

_DR = [-1, 1, 0, 0]
_DC = [0, 0, -1, 1]
_nb = np.full((N + 1, 4), N, np.int32)  # N 은 "벽" (보드 밖)
for _c in range(N):
    for _d in range(4):
        _r, _col = divmod(_c, 7)
        _rr, _cc = _r + _DR[_d], _col + _DC[_d]
        if 0 <= _rr < 7 and 0 <= _cc < 7:
            _nb[_c, _d] = _rr * 7 + _cc
NB = jnp.asarray(_nb)

_START = [[1, 1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 1], [0] * 7, [0] * 7, [0] * 7,
          [1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1]]


def _mask(cells):
    m = np.zeros(N, bool)
    m[list(cells)] = True
    return m


START_TILES = jnp.asarray(np.array(_START, bool).reshape(N))
START_P0 = jnp.asarray(_mask([0, 6]))
START_P1 = jnp.asarray(_mask([42, 48]))

_rng = np.random.RandomState(12345)
_KEYS = jnp.asarray(_rng.randint(1, 2**31 - 1, size=(3, N, 2)).astype(np.uint32) * 2 + 1)


def pos_hash(tiles, me, opp):
    """국면(타일, 둘 차례인 쪽 말, 상대 말) 해시. 32비트 둘."""
    parts = jnp.where(tiles[:, None], _KEYS[0], 0) ^ jnp.where(me[:, None], _KEYS[1], 0) ^ jnp.where(opp[:, None], _KEYS[2], 0)
    return jnp.bitwise_xor.reduce(parts, axis=0)


def slide_dest(tiles):
    """각 (칸, 방향)에서 이웃 빈 칸으로 미끄러졌을 때 멈추는 칸. 이웃이 빈 칸일 때만 의미가 있다."""
    tp = jnp.concatenate([tiles, jnp.array([True])])
    d_idx = jnp.arange(4)[None, :]

    def body(_, pos):
        nxt = NB[pos, d_idx]
        return jnp.where(~tp[nxt], nxt, pos)

    return lax.fori_loop(0, 6, body, NB[:N])


def reach(tiles, blocked, s):
    free = tiles & ~blocked
    vis = jnp.zeros(N, bool).at[s].set(True)

    def body(_, vis):
        vp = jnp.concatenate([vis, jnp.array([False])])
        dil = vp[NB[:N, 0]] | vp[NB[:N, 1]] | vp[NB[:N, 2]] | vp[NB[:N, 3]]
        return vis | (dil & free)

    return lax.fori_loop(0, 12, body, vis)


def legal(tiles, me, opp):
    """(가능한 수 bool[392], 말 칸 번호 int[2])."""
    occ = me | opp
    srcs = jnp.nonzero(me, size=2, fill_value=0)[0]
    tp = jnp.concatenate([tiles, jnp.array([True])])
    empty_nb = ~tp[NB[:N]]  # 이웃이 보드 안의 빈 칸

    def per(s):
        blocked = occ & ~(jnp.arange(N) == s)
        return reach(tiles, blocked, s)[:, None] & empty_nb

    return jax.vmap(per)(srcs).reshape(NUM_MOVES), srcs


def apply_move(tiles, me, srcs, idx):
    """수를 두고 (새 tiles, 새 me, 멈춘 칸) 을 돌려준다."""
    k, c, d = idx // (N * 4), (idx % (N * 4)) // 4, idx % 4
    dst = slide_dest(tiles)[c, d]
    s = srcs[k]
    return tiles.at[c].set(False).at[dst].set(True), me.at[s].set(False).at[dst].set(True), dst


def nth_true(valid, j):
    """valid 에서 j 번째(0부터) True 의 인덱스."""
    return jnp.argmax(jnp.cumsum(valid) > j)


def move_int(srcs, idx, dst):
    """Rust 의 수 정수 표현: from | dir<<6 | dest<<8 | src<<14."""
    k, c, d = idx // (N * 4), (idx % (N * 4)) // 4, idx % 4
    return c | (d << 6) | (dst << 8) | (srcs[k] << 14)


def _step(policy, carry, key, max_plies):
    """한 수를 둔다. 이미 끝난 판이면 아무것도 바꾸지 않는다 (기록 배열은 안 쓰는 칸에 쓰므로 통째로 복사하지 않는다)."""
    tiles, me, opp, hist, done, winner, ply = carry
    valid, srcs = legal(tiles, me, opp)
    n = valid.sum()
    has = n > 0
    j = policy(valid, n, ply, jax.random.fold_in(key, ply))
    idx = nth_true(valid, jnp.clip(j, 0, jnp.maximum(n - 1, 0)))
    t2, m2, dst = apply_move(tiles, me, srcs, idx)
    win = has & (dst == GOAL)
    nt = jnp.where(has, t2, tiles)
    nm = jnp.where(has, m2, me)
    h = pos_hash(nt, opp, nm)  # 다음 차례 쪽에서 본 국면
    prior = ((hist == h).all(-1) & (jnp.arange(max_plies + 1) <= ply)).sum()
    rep = prior + 1 >= 3
    hist = hist.at[jnp.where(done, max_plies, ply + 1)].set(h)

    def keep(old, new):
        return jnp.where(done, old, new)

    return (keep(tiles, nt), keep(me, opp), keep(opp, nm), hist, keep(done, win | rep),
            keep(winner, jnp.where(win, ply & 1, -1).astype(jnp.int32)), keep(ply, ply + 1))


def _initial(max_plies):
    hist = jnp.zeros((max_plies + 1, 2), jnp.uint32).at[0].set(pos_hash(START_TILES, START_P0, START_P1))
    return (START_TILES, START_P0, START_P1, hist, jnp.array(False), jnp.int32(-1), jnp.int32(0))


def play_game(policy, key, max_plies=DEFAULT_MAX_PLIES):
    """한 판을 끝까지 둔다. policy(valid, n, ply, key) -> 몇 번째 수를 둘지(0..n-1).
    (winner, plies, ended) 를 돌려준다. winner: -1 무승부, 0 먼저 둔 쪽, 1 나중에 둔 쪽.
    ended 가 False 면 max_plies 까지 가도 안 끝난 판이다 (무승부로 센다)."""
    final = lax.while_loop(lambda c: (~c[4]) & (c[6] < max_plies),
                           lambda c: _step(policy, c, key, max_plies), _initial(max_plies))
    return final[5], final[6], final[4]


def play_batch(policy, keys, max_plies=DEFAULT_MAX_PLIES):
    """판 여러 개를 동시에 둔다 (keys: 판마다 난수 키). 모두 끝나면 멈춘다. play_game 과 같은 결과를 더 빨리 낸다."""
    B = keys.shape[0]
    init = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), _initial(max_plies))
    step = jax.vmap(lambda c, k: _step(policy, c, k, max_plies))
    final = lax.while_loop(lambda c: (~c[4]).any() & (c[6].max() < max_plies),
                           lambda c: step(c, keys), init)
    return final[5], final[6], final[4]


def outcome_values(winner):
    """(먼저 둔 쪽 점수, 나중에 둔 쪽 점수). 이기면 1, 지면 0, 무승부는 둘 다 0 (둘 다 진 것)."""
    return (winner == 0).astype(jnp.float32), (winner == 1).astype(jnp.float32)


def random_policy(valid, n, ply, key):
    return jnp.minimum((jax.random.uniform(key) * n).astype(jnp.int32), n - 1)


def fixed_policy(a, b):
    """(ply*a + b) 번째 수. Rust 쪽 `rules_dump games` 와 같은 규칙."""
    return lambda valid, n, ply, key: (ply * a + b) % jnp.maximum(n, 1)


# ------------------------------------------------------------------ 순환 규칙 (지금 게임)
# 대각선 배치, 말 하나씩 시계 방향: 쪽0 (0,6) -> 쪽1 (6,6) -> 쪽0 (6,0) -> 쪽1 (0,0) -> ...
# 상태에 mc (둘 차례인 쪽이 지금 움직일 말, 칸 하나) 와 oc (상대가 다음 차례에 움직일 말) 를 더한다.
# 수 하나 = from*4 + dir (196개, 움직일 말이 정해져 있어서 말 번호가 필요 없다). 순서는 Rust generate_turn 과 같다.
NUM_TURN_MOVES = N * 4
CYCLE_P0 = jnp.asarray(_mask([6, 42]))
CYCLE_P1 = jnp.asarray(_mask([48, 0]))
CYCLE_C0 = jnp.asarray(_mask([6]))
CYCLE_C1 = jnp.asarray(_mask([48]))
_TKEYS = jnp.asarray(_rng.randint(1, 2**31 - 1, size=(2, N, 2)).astype(np.uint32) * 2 + 1)


def turn_hash(tiles, me, opp, mc, oc):
    """차례 정보(mc, oc)까지 넣은 국면 해시."""
    extra = jnp.where(mc[:, None], _TKEYS[0], 0) ^ jnp.where(oc[:, None], _TKEYS[1], 0)
    return pos_hash(tiles, me, opp) ^ jnp.bitwise_xor.reduce(extra, axis=0)


def legal_turn(tiles, me, opp, mc):
    """mc 말의 가능한 수 bool[196] (from*4+dir) 와 그 말의 칸 번호."""
    s = jnp.argmax(mc)
    tp = jnp.concatenate([tiles, jnp.array([True])])
    empty_nb = ~tp[NB[:N]]
    blocked = (me | opp) & ~(jnp.arange(N) == s)
    return (reach(tiles, blocked, s)[:, None] & empty_nb).reshape(NUM_TURN_MOVES), s


def apply_turn(tiles, me, s, a):
    """칸 s 의 말로 수 a(from*4+dir)를 둔다. (새 tiles, 새 me, 멈춘 칸)."""
    c, d = a // 4, a % 4
    dst = slide_dest(tiles)[c, d]
    return tiles.at[c].set(False).at[dst].set(True), me.at[s].set(False).at[dst].set(True), dst


def _cycle_step(policy, carry, key, max_plies):
    tiles, me, opp, mc, oc, hist, done, winner, ply = carry
    valid, s = legal_turn(tiles, me, opp, mc)
    n = valid.sum()
    has = n > 0
    j = policy(valid, n, ply, jax.random.fold_in(key, ply))
    a = nth_true(valid, jnp.clip(j, 0, jnp.maximum(n - 1, 0)))
    t2, m2, dst = apply_turn(tiles, me, s, a)
    win = has & (dst == GOAL)
    nt = jnp.where(has, t2, tiles)
    nm = jnp.where(has, m2, me)
    nc = me & ~mc  # 내 다음 차례에는 다른 말
    h = turn_hash(nt, opp, nm, oc, nc)
    prior = ((hist == h).all(-1) & (jnp.arange(max_plies + 1) <= ply)).sum()
    rep = prior + 1 >= 3
    hist = hist.at[jnp.where(done, max_plies, ply + 1)].set(h)

    def keep(old, new):
        return jnp.where(done, old, new)

    return (keep(tiles, nt), keep(me, opp), keep(opp, nm), keep(mc, oc), keep(oc, nc), hist, keep(done, win | rep),
            keep(winner, jnp.where(win, ply & 1, -1).astype(jnp.int32)), keep(ply, ply + 1))


def play_cycle_game(policy, key, max_plies=DEFAULT_MAX_PLIES):
    """순환 규칙으로 한 판. (winner, plies, ended)."""
    hist = jnp.zeros((max_plies + 1, 2), jnp.uint32).at[0].set(turn_hash(START_TILES, CYCLE_P0, CYCLE_P1, CYCLE_C0, CYCLE_C1))
    init = (START_TILES, CYCLE_P0, CYCLE_P1, CYCLE_C0, CYCLE_C1, hist, jnp.array(False), jnp.int32(-1), jnp.int32(0))
    final = lax.while_loop(lambda c: (~c[6]) & (c[8] < max_plies), lambda c: _cycle_step(policy, c, key, max_plies), init)
    return final[7], final[8], final[6]

"""Slider 2v2 규칙을 JAX 배열 연산으로 다시 쓴 것 (GPU/TPU/CPU 어디서나 돈다).

규칙은 rust/src/bitboard.rs 와 같다: 한 턴에 내 말 하나를 골라 타일 위를 걷고(다른 말이 있는 칸은 못 지남),
빈 칸 방향으로 서 있는 타일째 미끄러진다. 어느 말이든 골(3,3)에 멈추면 승리, 수가 없으면 통과,
같은 국면(차례 포함)이 3번 나오면 무승부, 150수가 넘으면 무승부.

상태는 칸 49개짜리 bool 배열 3개 (tiles, me=둘 차례인 쪽 말, opp). 수 하나 = 평평한 인덱스
k*196 + from*4 + dir  (k = 둘 중 몇 번째 말인지, 칸 번호 오름차순). 이 순서는 Rust 의 generate 순서와 같다.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

N = 49
GOAL = 24
MAX_PLIES = 150
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


def play_game(policy, key):
    """한 판을 끝까지 둔다. policy(valid, n, ply, key) -> 몇 번째 수를 둘지(0..n-1). (winner, plies) 를 돌려준다.
    winner: -1 무승부, 0 먼저 둔 쪽, 1 나중에 둔 쪽."""
    hist = jnp.zeros((MAX_PLIES + 1, 2), jnp.uint32).at[0].set(pos_hash(START_TILES, START_P0, START_P1))
    init = (START_TILES, START_P0, START_P1, hist, jnp.array(False), jnp.int32(-1), jnp.int32(0))

    def step(carry, ply):
        tiles, me, opp, hist, done, winner, plies = carry
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
        prior = ((hist == h).all(-1) & (jnp.arange(MAX_PLIES + 1) <= ply)).sum()
        rep = prior + 1 >= 3
        new = (nt, opp, nm, hist.at[ply + 1].set(h), win | rep, jnp.where(win, ply & 1, -1).astype(jnp.int32), plies + 1)
        out = jax.tree_util.tree_map(lambda o, n_: jnp.where(done, o, n_), carry, new)
        return out, None

    final, _ = lax.scan(step, init, jnp.arange(MAX_PLIES))
    return final[5], final[6]


def random_policy(valid, n, ply, key):
    return jnp.minimum((jax.random.uniform(key) * n).astype(jnp.int32), n - 1)


def fixed_policy(a, b):
    """(ply*a + b) 번째 수. Rust 쪽 `rules_dump games` 와 같은 규칙."""
    return lambda valid, n, ply, key: (ply * a + b) % jnp.maximum(n, 1)

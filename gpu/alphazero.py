"""Slider 강화학습 (알파제로 방식): 신경망(정책 + 가치) + MCTS 자가대전, 전부 JAX 로 GPU 에서.

- 규칙은 slider_jax.py (Rust 엔진과 규칙이 같음을 확인한 구현)를 쓴다.
- 수 393개 = 말(2) x 칸(49) x 방향(4) + 통과(둘 수 있는 수가 없을 때만).
- 신경망 입력: 타일 / 둘 차례인 쪽 말 / 상대 말 (49칸씩 147개). 출력: 수마다의 확률(정책), 이길 확률(가치, -1..1).
- 한 반복 = 판 B개를 동시에 MCTS(mctx 의 Gumbel MuZero 정책)로 끝까지 두고 -> 그 기록으로 신경망 학습.
- 승패: 이기면 +1, 지면 -1, 무승부(같은 국면 3번, 또는 MAX_LEN 수) 0.
"""
import functools
import pickle
import time

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax

import slider_jax as sj

A = sj.NUM_MOVES + 1  # 마지막 = 통과
PASS = sj.NUM_MOVES
OBS = 3 * sj.N


# ------------------------------------------------------------------ 게임
def legal_mask(tiles, me, opp):
    valid, _ = sj.legal(tiles, me, opp)
    return jnp.concatenate([valid, (~valid.any())[None]])


def observe(tiles, me, opp):
    return jnp.concatenate([tiles, me, opp]).astype(jnp.float32)


def step(tiles, me, opp, action):
    """수를 둔다. (다음 차례 기준 tiles, me, opp, 방금 둔 쪽의 보상, 이겼는지)."""
    _, srcs = sj.legal(tiles, me, opp)
    is_pass = action == PASS
    t2, m2, dst = sj.apply_move(tiles, me, srcs, jnp.where(is_pass, 0, action))
    win = (~is_pass) & (dst == sj.GOAL)
    nt = jnp.where(is_pass, tiles, t2)
    nm = jnp.where(is_pass, me, m2)
    return nt, opp, nm, win.astype(jnp.float32), win


# ------------------------------------------------------------------ 신경망 (MLP)
def init_params(key, hidden=256):
    def dense(k, n_in, n_out, scale=None):
        scale = np.sqrt(2.0 / n_in) if scale is None else scale
        return {"w": jax.random.normal(k, (n_in, n_out)) * scale, "b": jnp.zeros(n_out)}

    k = jax.random.split(key, 5)
    return {"l1": dense(k[0], OBS, hidden), "l2": dense(k[1], hidden, hidden), "l3": dense(k[2], hidden, hidden),
            "policy": dense(k[3], hidden, A, 0.01), "value": dense(k[4], hidden, 1, 0.01)}


def apply_net(params, x):
    h = jax.nn.relu(x @ params["l1"]["w"] + params["l1"]["b"])
    h = h + jax.nn.relu(h @ params["l2"]["w"] + params["l2"]["b"])
    h = h + jax.nn.relu(h @ params["l3"]["w"] + params["l3"]["b"])
    logits = h @ params["policy"]["w"] + params["policy"]["b"]
    value = jnp.tanh(h @ params["value"]["w"] + params["value"]["b"])[..., 0]
    return logits, value


def masked_logits(logits, mask):
    return jnp.where(mask, logits, -1e9)


# ------------------------------------------------------------------ MCTS
def recurrent_fn(params, rng, action, emb):
    tiles, me, opp, done = emb
    nt, nme, nopp, reward, win = jax.vmap(step)(tiles, me, opp, action)
    new_done = done | win
    reward = jnp.where(done, 0.0, reward)  # 이미 끝난 국면 뒤로는 보상 없음
    logits, value = apply_net(params, jax.vmap(observe)(nt, nme, nopp))
    logits = masked_logits(logits, jax.vmap(legal_mask)(nt, nme, nopp))
    value = jnp.where(new_done, 0.0, value)
    discount = jnp.where(new_done, 0.0, -1.0)  # 두 사람이 번갈아 두므로 가치의 부호가 바뀐다
    out = mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value)
    return out, (nt, nme, nopp, new_done)


def run_mcts(params, rng, tiles, me, opp, num_simulations, max_considered=16, invalid=None):
    mask = jax.vmap(legal_mask)(tiles, me, opp) if invalid is None else ~invalid
    logits, value = apply_net(params, jax.vmap(observe)(tiles, me, opp))
    done = jnp.zeros(tiles.shape[0], bool)
    root = mctx.RootFnOutput(prior_logits=masked_logits(logits, mask), value=value, embedding=(tiles, me, opp, done))
    return mctx.gumbel_muzero_policy(params, rng, root, recurrent_fn, num_simulations=num_simulations,
                                     invalid_actions=~mask, max_num_considered_actions=max_considered)


# ------------------------------------------------------------------ 형세에 따른 무승부 판단 (실제 판의 3회 반복)
def draw_moves(tiles, me, opp, hist, ply, max_len):
    """수마다: 그 수를 두면 같은 국면이 3번째가 되어 무승부로 끝나는지 [A]. (MCTS 는 실제 판의 기록을 모르므로 루트에서 따로 본다)"""
    _, srcs = sj.legal(tiles, me, opp)

    def after(a):
        t2, m2, dst = sj.apply_move(tiles, me, srcs, a)
        return sj.pos_hash(t2, opp, m2), dst == sj.GOAL

    hs, wins = jax.vmap(after)(jnp.arange(sj.NUM_MOVES))
    hs = jnp.concatenate([hs, sj.pos_hash(tiles, opp, me)[None]], 0)  # 통과
    wins = jnp.concatenate([wins, jnp.array([False])])
    upto = jnp.arange(max_len + 1) <= ply
    counts = ((hist[None, :, :] == hs[:, None, :]).all(-1) & upto[None]).sum(-1)
    return (counts >= 2) & ~wins


def choose_action(params, key, tiles, me, opp, hist, ply, num_simulations, max_len, draw_threshold):
    """MCTS 로 수를 고른다. draw_threshold 가 있으면: 이기는 중(가치 > 기준)이면 무승부가 되는 수를 빼고,
    지는 중(가치 < -기준)이면 무승부가 되는 수가 있을 때 그 수를 둔다. 비슷하면 그대로."""
    if draw_threshold is None:
        out = run_mcts(params, key, tiles, me, opp, num_simulations)
        return out.action, out
    mask = jax.vmap(legal_mask)(tiles, me, opp)
    dm = jax.vmap(draw_moves, in_axes=(0, 0, 0, 0, None, None))(tiles, me, opp, hist, ply, max_len) & mask
    _, v = apply_net(params, jax.vmap(observe)(tiles, me, opp))
    avoid = (v > draw_threshold) & (mask & ~dm).any(-1)
    out = run_mcts(params, key, tiles, me, opp, num_simulations, invalid=~mask | (avoid[:, None] & dm))
    take_draw = (v < -draw_threshold) & dm.any(-1)
    return jnp.where(take_draw, jnp.argmax(dm, -1), out.action), out


# ------------------------------------------------------------------ 판 진행 (B 판 동시, 최대 max_len 수)
def _start(batch, max_len):
    tiles = jnp.broadcast_to(sj.START_TILES, (batch, sj.N))
    p0 = jnp.broadcast_to(sj.START_P0, (batch, sj.N))
    p1 = jnp.broadcast_to(sj.START_P1, (batch, sj.N))
    h0 = sj.pos_hash(sj.START_TILES, sj.START_P0, sj.START_P1)
    hist = jnp.zeros((batch, max_len + 1, 2), jnp.uint32).at[:, 0].set(h0)
    return tiles, p0, p1, hist


def _advance(tiles, me, opp, hist, done, winner, ply, action, max_len):
    """한 수 진행 + 승리 / 3회 반복 판정. ply 는 모든 판이 같다 (판마다 처음부터 같이 시작)."""
    nt, nme, nopp, _, win = jax.vmap(step)(tiles, me, opp, action)
    h = jax.vmap(sj.pos_hash)(nt, nme, nopp)
    seen = ((hist == h[:, None, :]).all(-1) & (jnp.arange(max_len + 1) <= ply)[None]).sum(-1)
    rep = seen + 1 >= 3
    hist = hist.at[:, ply + 1].set(h)
    keep = lambda old, new: jnp.where(done.reshape((-1,) + (1,) * (new.ndim - 1)), old, new)
    winner = jnp.where(done, winner, jnp.where(win, ply & 1, -1))
    return keep(tiles, nt), keep(me, nme), keep(opp, nopp), hist, done | win | rep, winner


@functools.partial(jax.jit, static_argnames=("batch", "num_simulations", "max_len", "draw_threshold"))
def self_play(params, rng, batch, num_simulations, max_len, draw_threshold=None):
    """B 판을 끝까지 둔다. 반환: 입력[T,B,147], 정책 목표[T,B,A], 그 수에서 둘 차례였는지[T,B], 승자[B] (-1 무승부, 0 먼저, 1 나중)."""
    tiles, me, opp, hist = _start(batch, max_len)
    init = (tiles, me, opp, hist, jnp.zeros(batch, bool), jnp.full(batch, -1, jnp.int32))

    def body(carry, inp):
        tiles, me, opp, hist, done, winner = carry
        ply, key = inp
        action, out = choose_action(params, key, tiles, me, opp, hist, ply, num_simulations, max_len, draw_threshold)
        rec = (jax.vmap(observe)(tiles, me, opp), out.action_weights, ~done)
        new = _advance(tiles, me, opp, hist, done, winner, ply, action, max_len)
        return new, rec

    keys = jax.random.split(rng, max_len)
    final, (obs, pi, active) = jax.lax.scan(body, init, (jnp.arange(max_len), keys))
    return obs, pi, active, final[5]


def value_targets(active, winner):
    """수마다 둘 차례였던 쪽에서 본 결과: 이김 +1, 짐 -1, 무승부 0."""
    T = active.shape[0]
    mover = (np.arange(T) % 2)[:, None]
    z = np.where(winner[None] < 0, 0.0, np.where(winner[None] == mover, 1.0, -1.0))
    return z.astype(np.float32)


# ------------------------------------------------------------------ 학습
def make_optimizer(lr):
    return optax.adamw(lr, weight_decay=1e-4)


def loss_fn(params, obs, pi, z):
    logits, v = apply_net(params, obs)
    policy_loss = optax.softmax_cross_entropy(logits, pi).mean()
    value_loss = jnp.mean((v - z) ** 2)
    return policy_loss + value_loss, (policy_loss, value_loss)


@functools.partial(jax.jit, static_argnames=("opt",))
def train_step(params, opt_state, obs, pi, z, opt):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, obs, pi, z)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


# ------------------------------------------------------------------ 평가 상대
def greedy_policy(tiles, me, opp, key):
    """간단한 전술 상대: 바로 이기는 수가 있으면 두고, 없으면 상대에게 바로 이기는 수를 주지 않는 수 중에서 무작위."""
    valid, srcs = sj.legal(tiles, me, opp)

    cells = (jnp.arange(sj.NUM_MOVES) % (sj.N * 4)) // 4
    dirs = jnp.arange(sj.NUM_MOVES) % 4

    def outcome(a):
        t2, m2, dst = sj.apply_move(tiles, me, srcs, a)
        ov, _ = sj.legal(t2, opp, m2)
        opp_wins = (ov & (sj.slide_dest(t2)[cells, dirs] == sj.GOAL)).any()
        return dst == sj.GOAL, opp_wins

    wins, gives = jax.vmap(outcome)(jnp.arange(sj.NUM_MOVES))
    noise = jax.random.uniform(key, (sj.NUM_MOVES,))
    score = jnp.where(valid, noise + 2.0 * (wins & valid) + 1.0 * (~gives & valid), -1.0)
    a = jnp.argmax(score)
    return jnp.where(valid.any(), a, PASS)


def random_policy(tiles, me, opp, key):
    valid, _ = sj.legal(tiles, me, opp)
    score = jnp.where(valid, jax.random.uniform(key, (sj.NUM_MOVES,)), -1.0)
    return jnp.where(valid.any(), jnp.argmax(score), PASS)


@functools.partial(jax.jit, static_argnames=("batch", "num_simulations", "max_len", "opponent", "draw_threshold"))
def evaluate(params, rng, batch, num_simulations, max_len, opponent, draw_threshold=None):
    """신경망+MCTS 가 판의 절반은 먼저, 절반은 나중에 둔다. 반환: (이긴 판, 무승부, 진 판)."""
    opp_policy = greedy_policy if opponent == "greedy" else random_policy
    tiles, me, opp, hist = _start(batch, max_len)
    agent_first = jnp.arange(batch) % 2 == 0
    init = (tiles, me, opp, hist, jnp.zeros(batch, bool), jnp.full(batch, -1, jnp.int32))

    def body(carry, inp):
        tiles, me, opp, hist, done, winner = carry
        ply, key = inp
        k1, k2 = jax.random.split(key)
        agent_turn = agent_first == (ply % 2 == 0)
        mcts_action, _ = choose_action(params, k1, tiles, me, opp, hist, ply, num_simulations, max_len, draw_threshold)
        other = jax.vmap(opp_policy)(tiles, me, opp, jax.random.split(k2, batch))
        action = jnp.where(agent_turn, mcts_action, other)
        return _advance(tiles, me, opp, hist, done, winner, ply, action, max_len), None

    final, _ = jax.lax.scan(body, init, (jnp.arange(max_len), jax.random.split(rng, max_len)))
    winner = final[5]
    agent_side = jnp.where(agent_first, 0, 1)
    return (winner == agent_side).sum(), (winner < 0).sum(), ((winner >= 0) & (winner != agent_side)).sum()


# ------------------------------------------------------------------ 저장
def save(path, params, opt_state, it, history):
    with open(path, "wb") as f:
        pickle.dump({"params": jax.device_get(params), "opt_state": jax.device_get(opt_state), "iteration": it,
                     "history": history}, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

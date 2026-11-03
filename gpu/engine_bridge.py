"""Rust 엔진(알파베타, VM 에서 검증한 설정)과 강화학습(JAX)을 잇는 코드.

- load_engine_data: Rust `selfplay --positions yes` 로 만든 판(엔진이 고른 수 + 판 결과)을 신경망 학습용 배열로
- pretrain: 그 데이터로 정책(엔진이 고른 수)과 가치(판 결과)를 먼저 배운다
- Engine: Rust `bestmove` 서버. 국면을 모아 보내면 엔진의 수를 돌려준다
- evaluate_vs_engine: 신경망+MCTS 와 Rust 엔진의 대국
"""
import json
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import alphazero as az

_SHIFTS = np.arange(49, dtype=np.uint64)
_POW = (np.uint64(1) << _SHIFTS)


def bits_to_mask(x):
    return ((np.uint64(x) >> _SHIFTS) & np.uint64(1)).astype(bool)


def mask_to_bits(m):
    return int(_POW[np.asarray(m, bool)].sum())


def rust_move_to_action(m, me_mask):
    """Rust 수 정수(from | dir<<6 | dest<<8 | src<<14) -> 강화학습의 수 번호 (말 순서는 칸 번호 오름차순)."""
    if m < 0:
        return az.PASS
    frm, d, src = m & 63, (m >> 6) & 3, (m >> 14) & 63
    cells = np.nonzero(me_mask)[0]
    k = 0 if src == cells[0] else 1
    return int(k * 196 + frm * 4 + d)


def load_engine_data(paths):
    """(입력 uint8[N,147], 수 번호 int32[N], 가치 목표 float32[N]: 이김 +1 / 무 0 / 짐 -1)."""
    obs, act, z = [], [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if "move" not in r:
                    continue
                t, me, op = bits_to_mask(r["tiles"]), bits_to_mask(r["me"]), bits_to_mask(r["opp"])
                obs.append(np.concatenate([t, me, op]).astype(np.uint8))
                act.append(rust_move_to_action(r["move"], me))
                z.append(2.0 * r["outcome"] - 1.0)
    return np.stack(obs), np.array(act, np.int32), np.array(z, np.float32)


def pretrain(params, opt_state, opt, obs, act, z, steps, batch=1024, seed=0, log_every=500):
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(steps):
        idx = rng.integers(0, len(z), size=min(batch, len(z)))
        pi = jax.nn.one_hot(jnp.asarray(act[idx]), az.A)
        params, opt_state, loss, (pl, vl) = az.train_step(params, opt_state, jnp.asarray(obs[idx], jnp.float32), pi,
                                                          jnp.asarray(z[idx]), opt)
        hist.append((float(pl), float(vl)))
        if (s + 1) % log_every == 0:
            p, v = np.mean(hist[-log_every:], axis=0)
            print(f"  사전학습 {s + 1}/{steps}: 정책 손실 {p:.3f}, 가치 손실 {v:.3f}", flush=True)
    return params, opt_state


def policy_accuracy(params, obs, act, n=20000, seed=0):
    """엔진이 고른 수를 신경망의 가장 높은 확률 수로 맞히는 비율 (가능한 수 안에서)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(act), size=min(n, len(act)), replace=False)
    x = jnp.asarray(obs[idx], jnp.float32)
    logits, _ = az.apply_net(params, x)
    t, me, op = x[:, :49] > 0, x[:, 49:98] > 0, x[:, 98:] > 0
    mask = jax.vmap(az.legal_mask)(t, me, op)
    pred = np.asarray(jnp.argmax(jnp.where(mask, logits, -1e9), -1))
    return float((pred == act[idx]).mean())


class Engine:
    """Rust bestmove 서버. moves([(tiles, me, opp), ...]) -> 엔진의 수 정수 목록 (-1 = 통과).
    서버가 죽었거나(셀을 중간에 멈춰서) 주고받기가 꼬였으면 새로 띄워서 다시 묻는다."""

    def __init__(self, binary, time_ms=50, threads=None, search="lmr2", weights="tuned", err_path="/tmp/bestmove_err.txt"):
        self.args = [str(binary), "--time-ms", str(time_ms), "--search", search, "--weights", weights]
        if threads:
            self.args += ["--threads", str(threads)]
        self.err_path = err_path
        self.p = None
        self.dirty = False
        self._start()

    def _start(self):
        if self.p is not None and self.p.poll() is None:
            self.p.kill()
            self.p.wait()
        self.err = open(self.err_path, "a")
        self.p = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1)
        self.dirty = False

    def _ask(self, positions):
        self.dirty = True
        for t, m, o in positions:
            self.p.stdin.write(f"{t} {m} {o}\n")
        self.p.stdin.write("\n")
        self.p.stdin.flush()
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("bestmove 서버가 끝났습니다")
            if line.strip() == "":
                break
            out.append(int(line.split()[0]))
        if len(out) != len(positions):
            raise RuntimeError(f"답 개수가 다릅니다: {len(out)} / {len(positions)}")
        self.dirty = False
        return out

    def moves(self, positions):
        if self.dirty or self.p.poll() is not None:
            self._start()
        try:
            return self._ask(positions)
        except (BrokenPipeError, RuntimeError, ValueError) as e:
            try:
                tail = open(self.err_path).read()[-500:]
            except OSError:
                tail = ""
            print(f"(엔진 서버를 다시 띄웁니다: {e}) {tail}", flush=True)
            self._start()
            return self._ask(positions)

    def close(self):
        if self.p is not None and self.p.poll() is None:
            self.p.stdin.close()
            self.p.wait(timeout=10)


def evaluate_vs_engine(params, engine, batch, num_simulations, max_len, draw_threshold=None, seed=0):
    """신경망+MCTS 가 판의 절반은 먼저, 절반은 나중에 Rust 엔진과 둔다. (이긴 판, 무승부, 진 판)."""
    choose = jax.jit(lambda p, k, t, m, o, h, ply: az.choose_action(p, k, t, m, o, h, ply, num_simulations, max_len,
                                                                     draw_threshold)[0])
    advance = jax.jit(az._advance, static_argnames=("max_len",))
    tiles, me, opp, hist = az._start(batch, max_len)
    done = jnp.zeros(batch, bool)
    winner = jnp.full(batch, -1, jnp.int32)
    agent_first = np.arange(batch) % 2 == 0
    key = jax.random.PRNGKey(seed)
    for ply in range(max_len):
        d = np.asarray(done)
        if d.all():
            break
        key, k = jax.random.split(key)
        action = np.array(choose(params, k, tiles, me, opp, hist, jnp.int32(ply)))
        engine_turn = (agent_first != (ply % 2 == 0)) & ~d
        need = np.nonzero(engine_turn)[0]
        if len(need):
            T, M, O = np.asarray(tiles), np.asarray(me), np.asarray(opp)
            em = engine.moves([(mask_to_bits(T[i]), mask_to_bits(M[i]), mask_to_bits(O[i])) for i in need])
            for i, m in zip(need, em):
                action[i] = rust_move_to_action(m, M[i])
        tiles, me, opp, hist, done, winner = advance(tiles, me, opp, hist, done, winner, jnp.int32(ply),
                                                     jnp.asarray(action), max_len=max_len)
    w = np.asarray(winner)
    agent_side = np.where(agent_first, 0, 1)
    return int((w == agent_side).sum()), int((w < 0).sum()), int(((w >= 0) & (w != agent_side)).sum())

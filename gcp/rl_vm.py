"""GPU VM 에서 강화학습 (colab/slider_alphazero_conv.ipynb 와 같은 과정, Drive 대신 VM 디스크와 guest attribute).

1. 엔진의 판 (Rust selfplay) -> 2. 사전학습 -> 3. 강화학습 (RL_HOURS 시간 동안) -> 결과 요약.
끊겼다가 다시 실행하면 저장 파일에서 이어서 한다. 설정은 환경변수로 바꿀 수 있다.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import alphazero as az  # noqa: E402
import engine_bridge as eb  # noqa: E402

E = os.environ.get
ARCH = E("ARCH", "conv")
CHANNELS = int(E("CHANNELS", 64))
BLOCKS = int(E("BLOCKS", 6))
HIDDEN = int(E("HIDDEN", 256))
ENGINE_SETS = [(600, 1), (1400, 2)]
ENGINE_TIME_MS = 30
PRETRAIN_STEPS = int(E("PRETRAIN_STEPS", 12000))
GAMES = int(E("GAMES", 512))
SIMULATIONS = int(E("SIMULATIONS", 32))
MAX_LEN = 200
DRAW_THRESHOLD = 0.2
LR = 1e-3
TRAIN_STEPS = 200
MINIBATCH = 1024
ENGINE_MIX = 0.25
REPLAY = 8
RL_HOURS = float(E("RL_HOURS", 4))
EVAL_EVERY = 5
EVAL_GAMES = 256
ENGINE_EVAL_EVERY = 10
ENGINE_EVAL_GAMES = 64
ENGINE_EVAL_TIME_MS = 20
SEED = 0
OUT = Path(E("OUT", Path.home() / "slider_rl"))
OUT.mkdir(parents=True, exist_ok=True)
BIN = REPO / "rust" / "target" / "release"
CPUS = os.cpu_count()
TAG = f"conv_c{CHANNELS}b{BLOCKS}" if ARCH == "conv" else f"mlp_h{HIDDEN}"


def put_attr(key, value):
    """VM 밖에서 읽을 수 있게 guest attribute 로 남긴다 (VM 이 아니면 무시)."""
    try:
        req = urllib.request.Request(
            f"http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/slider/{key}",
            data=str(value)[:200000].encode(), method="PUT", headers={"Metadata-Flavor": "Google"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def sync():
    """저장 파일을 GCS 로 올린다 (BUCKET 이 있을 때만). VM 이 지워져도 남게."""
    b = E("BUCKET")
    if b:
        files = [str(f) for f in OUT.glob("*") if f.suffix in (".pkl", ".json", ".txt")]
        subprocess.run(["gsutil", "-q", "-m", "cp", *files, f"gs://{b}/"], check=False)


def log(msg):
    print(msg, flush=True)
    with open(OUT / "log.txt", "a") as f:
        f.write(msg + "\n")


def main():
    log(f"device: {jax.devices()} | CPU {CPUS} | {TAG}")
    eng_files = []
    for n_games, seed in ENGINE_SETS:
        path = OUT / f"engine_{n_games}g_{ENGINE_TIME_MS}ms_s{seed}.jsonl"
        if not path.exists():
            t0 = time.time()
            tmp = path.with_suffix(".tmp")
            subprocess.run([str(BIN / "selfplay"), "--games", str(n_games), "--time-ms", str(ENGINE_TIME_MS), "--opts", "lmr2",
                            "--weights", str(REPO / "weights" / "tuned.json"), "--max-plies", "1000", "--random-plies", "4",
                            "--epsilon", "0.05", "--threads", str(CPUS), "--seed", str(seed), "--positions", "yes",
                            "--out", str(tmp)], check=True, stdout=subprocess.DEVNULL)
            tmp.rename(path)
            log(f"엔진 판 {n_games}개 만듦 ({time.time() - t0:.0f}초)")
        eng_files.append(str(path))
    E_obs, E_act, E_z = eb.load_engine_data(eng_files)
    log(f"엔진 판 국면: {len(E_act)}")

    ckpt = OUT / f"ckpt_{TAG}.pkl"
    hist_path = OUT / f"history_{TAG}.json"
    opt = az.make_optimizer(LR)
    engine = eb.Engine(BIN / "bestmove", time_ms=ENGINE_EVAL_TIME_MS, threads=CPUS)
    if ckpt.exists():
        ck = az.load(str(ckpt))
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        opt_state = jax.tree_util.tree_map(jnp.asarray, ck["opt_state"])
        start_it, history = ck["iteration"], ck["history"]
        log(f"이어서 학습: 반복 {start_it} 부터")
    else:
        params = az.make_params(jax.random.PRNGKey(SEED), arch=ARCH, hidden=HIDDEN, channels=CHANNELS, blocks=BLOCKS)
        opt_state = opt.init(params)
        log(f"신경망 크기 {sum(x.size for x in jax.tree_util.tree_leaves(params))}, 엔진의 판으로 사전학습 {PRETRAIN_STEPS}단계")
        t0 = time.time()
        params, opt_state = eb.pretrain(params, opt_state, opt, E_obs, E_act, E_z, PRETRAIN_STEPS, MINIBATCH, SEED)
        acc = eb.policy_accuracy(params, E_obs, E_act)
        w, d, l = eb.evaluate_vs_engine(params, engine, ENGINE_EVAL_GAMES, SIMULATIONS, MAX_LEN, DRAW_THRESHOLD, SEED)
        score = (w + 0.5 * d) / ENGINE_EVAL_GAMES
        log(f"사전학습 끝 ({time.time() - t0:.0f}초): 엔진 수 맞히기 {acc:.1%} | 엔진 상대 {w}승 {d}무 {l}패 ({score:.2f})")
        start_it, history = 0, [{"iteration": 0, "pretrain_acc": acc, "vs_engine": [w, d, l], "vs_engine_score": score}]
        az.save(str(ckpt), params, opt_state, 0, history)
    put_attr("progress", f"pretrain acc {history[0].get('pretrain_acc', 0):.3f}, vs_engine {history[0].get('vs_engine')}")

    rng = np.random.default_rng(SEED + start_it)
    key = jax.random.PRNGKey(SEED + 1000 + start_it)
    replay = []
    deadline = time.time() + RL_HOURS * 3600
    it = start_it
    while time.time() < deadline:
        t0 = time.time()
        key, k_play, k_eval = jax.random.split(key, 3)
        obs, pi, active, winner = az.self_play(params, k_play, batch=GAMES, num_simulations=SIMULATIONS, max_len=MAX_LEN,
                                               draw_threshold=DRAW_THRESHOLD)
        obs, pi, active, winner = (np.asarray(x) for x in (obs, pi, active, winner))
        z = az.value_targets(active, winner)
        replay.append((obs[active].astype(np.uint8), pi[active].astype(np.float16), z[active]))
        replay = replay[-REPLAY:]
        X = np.concatenate([r[0] for r in replay])
        P = np.concatenate([r[1] for r in replay])
        Z = np.concatenate([r[2] for r in replay])
        n_eng = int(MINIBATCH * ENGINE_MIX)
        losses = []
        for _ in range(TRAIN_STEPS):
            i = rng.integers(0, len(Z), size=MINIBATCH - n_eng)
            j = rng.integers(0, len(E_z), size=n_eng)
            xb = jnp.asarray(np.concatenate([X[i], E_obs[j]]), jnp.float32)
            pb = jnp.concatenate([jnp.asarray(P[i], jnp.float32), jax.nn.one_hot(jnp.asarray(E_act[j]), az.A)])
            zb = jnp.asarray(np.concatenate([Z[i], E_z[j]]))
            params, opt_state, loss, (pl, vl) = az.train_step(params, opt_state, xb, pb, zb, opt)
            losses.append((float(pl), float(vl)))
        it += 1
        pl, vl = np.mean(losses, axis=0)
        row = {"iteration": it, "avg_len": float(active.sum(0).mean()), "first_win": float((winner == 0).mean()),
               "second_win": float((winner == 1).mean()), "draw": float((winner < 0).mean()), "policy_loss": float(pl),
               "value_loss": float(vl)}
        if it % EVAL_EVERY == 0:
            w, d, l = (int(x) for x in az.evaluate(params, k_eval, batch=EVAL_GAMES, num_simulations=SIMULATIONS, max_len=MAX_LEN,
                                                   opponent="greedy", draw_threshold=DRAW_THRESHOLD))
            row["vs_greedy"], row["vs_greedy_score"] = [w, d, l], (w + 0.5 * d) / EVAL_GAMES
        if it % ENGINE_EVAL_EVERY == 0:
            w, d, l = eb.evaluate_vs_engine(params, engine, ENGINE_EVAL_GAMES, SIMULATIONS, MAX_LEN, DRAW_THRESHOLD, SEED + it)
            row["vs_engine"], row["vs_engine_score"] = [w, d, l], (w + 0.5 * d) / ENGINE_EVAL_GAMES
        row["sec"] = time.time() - t0
        history.append(row)
        msg = (f"[{it}] {row['sec']:.0f}s | 평균 {row['avg_len']:.0f}수, 선승 {row['first_win']:.0%} 후승 {row['second_win']:.0%} "
               f"무 {row['draw']:.0%} | 손실 정책 {pl:.3f} 가치 {vl:.3f}")
        if "vs_greedy" in row:
            msg += f" | 전술 상대 {row['vs_greedy']} ({row['vs_greedy_score']:.2f})"
        if "vs_engine" in row:
            msg += f" | Rust 엔진 {row['vs_engine']} ({row['vs_engine_score']:.2f})"
        log(msg)
        az.save(str(ckpt), params, opt_state, it, history)
        hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=1))
        if "vs_engine" in row or it % EVAL_EVERY == 0:
            put_attr("progress", msg)
        if "vs_engine" in row:
            sync()
    engine.close()
    lines = [l for l in (OUT / "log.txt").read_text().splitlines() if "엔진 상대" in l or "Rust 엔진" in l or "사전학습" in l]
    put_attr("summary", "\n".join(lines[-80:]))
    sync()


if __name__ == "__main__":
    main()

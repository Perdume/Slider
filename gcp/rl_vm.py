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
ENGINE_SETS = [(1000, 1), (1000, 2)]
ENGINE_TIME_MS = 30
PRETRAIN_STEPS = int(E("PRETRAIN_STEPS", 12000))
GAMES = int(E("GAMES", 512))
SIMULATIONS = int(E("SIMULATIONS", 32))
MAX_LEN = int(E("MAX_LEN", 3000))  # 평가(실제 규칙, 3회 반복 무승부) 판의 안전 한도. 자가대전은 수 제한 없음
MAX_AGE = int(E("MAX_AGE", 3000))  # 자가대전 안전장치: 이 수를 넘긴 판은 학습에서 빼고 새 판 (판은 승 / 3회 반복 무승부로 끝남)
GAMMA = float(E("GAMMA", 0.995))  # 이기는 보상을 남은 수만큼 할인 (빨리 이길수록 큼)
# 역방향 커리큘럼: 판의 CURR_FRAC 은 엔진 판의 막판 국면(끝까지 CURR_K 수 이하)에서 시작.
# 그 판들에서 이길 쪽이 실제로 이긴 비율이 CURR_PASS 이상이면 CURR_K 를 CURR_STEP 만큼 늘린다.
CURR_FRAC = float(E("CURR_FRAC", 0.5))
CURR_START = int(E("CURR_START", 4))
CURR_STEP = int(E("CURR_STEP", 4))
CURR_PASS = float(E("CURR_PASS", 0.75))
CURR_MAX = int(E("CURR_MAX", 400))
# 멈춤 기준: 이어 시작한 뒤 STOP_AFTER 반복 동안 엔진 상대 최고 점수가 STOP_SCORE 미만이면 끝낸다
STOP_AFTER = int(E("STOP_AFTER", 30))
STOP_SCORE = float(E("STOP_SCORE", 0.2))
DRAW_THRESHOLD = 0.2
LR = 1e-3
TRAIN_STEPS = 200
MINIBATCH = 1024
ENGINE_MIX = 0.25
REPLAY = 8
RL_HOURS = float(E("RL_HOURS", 4))
RL_UNTIL = float(E("RL_UNTIL", 0))  # 이 시각(유닉스 초)이 되면 학습을 멈춘다 (0 이면 RL_HOURS 만)
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
TAG = (f"cycle_conv_c{CHANNELS}b{BLOCKS}" if ARCH == "conv" else f"cycle_mlp_h{HIDDEN}")


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
        path = OUT / f"engine_cycle_{n_games}g_{ENGINE_TIME_MS}ms_s{seed}.jsonl"
        if not path.exists() and E("BUCKET"):
            # 다른 VM 이 미리 만들어 둔 판이 있으면 받는다
            subprocess.run(["gsutil", "-q", "cp", f"gs://{E('BUCKET')}/{path.name}", str(path)], check=False)
            if path.exists():
                log(f"엔진 판 받음: {path.name}")
        if not path.exists():
            t0 = time.time()
            tmp = path.with_suffix(".tmp")
            subprocess.run([str(BIN / "selfplay"), "--games", str(n_games), "--time-ms", str(ENGINE_TIME_MS), "--rules", "cycle", "--opts", "lmr2",
                            "--weights", str(REPO / "weights" / "tuned.json"), "--max-plies", "1000", "--random-plies", "4",
                            "--epsilon", "0.05", "--threads", str(CPUS), "--seed", str(seed), "--positions", "yes",
                            "--out", str(tmp)], check=True, stdout=subprocess.DEVNULL)
            tmp.rename(path)
            log(f"엔진 판 {n_games}개 만듦 ({time.time() - t0:.0f}초)")
        eng_files.append(str(path))
    az.SEARCH_GAMMA = GAMMA
    E_obs, E_act, E_z, pool = eb.load_engine_data(eng_files, gamma=GAMMA, with_starts=True)
    log(f"엔진 판 국면: {len(E_act)} | 커리큘럼 시작 국면 {len(pool['z'])} | 할인 {GAMMA} | 자가대전 수 제한 없음, 3회 반복은 무승부 (안전장치 {MAX_AGE}수)")

    ckpt = OUT / f"ckpt_{TAG}.pkl"
    hist_path = OUT / f"history_{TAG}.json"
    opt = az.make_optimizer(LR)
    engine = eb.Engine(BIN / "bestmove", time_ms=ENGINE_EVAL_TIME_MS, threads=CPUS)
    if not ckpt.exists() and E("BUCKET"):
        # 앞 VM 이 GCS 에 남긴 저장 파일이 있으면 받아서 이어서 한다
        for f in (ckpt, hist_path, OUT / "log.txt"):
            subprocess.run(["gsutil", "-q", "cp", f"gs://{E('BUCKET')}/{f.name}", str(f)], check=False)
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
    if RL_UNTIL > 0:
        deadline = min(deadline, RL_UNTIL)
    it = start_it
    curr_k = next((h["curr_k"] for h in reversed(history) if "curr_k" in h), CURR_START)
    import slider_jax as sj
    base = [np.asarray(x, bool) for x in (sj.START_TILES, sj.CYCLE_P0, sj.CYCLE_P1, sj.CYCLE_C0, sj.CYCLE_C1)]
    ck_state = {"k": curr_k}

    def sampler(n):
        """새 판의 시작 국면: CURR_FRAC 확률로 엔진 판의 막판 국면 (끝까지 curr_k 수 이하), 아니면 처음 배치.
        info = 막판 국면이면 이길 쪽 (0 = 시작 국면에서 둘 차례인 쪽), 처음 배치면 None."""
        out = [np.repeat(b[None], n, 0) for b in base]
        info = [None] * n
        cand = np.nonzero((pool["remaining"] >= 1) & (pool["remaining"] <= ck_state["k"]))[0]
        for i in range(n):
            if rng.random() < CURR_FRAC and len(cand):
                r = cand[rng.integers(len(cand))]
                for j in range(5):
                    out[j][i] = pool["state"][j][r]
                info[i] = 0 if pool["z"][r] > 0 else 1
        return tuple(out), info

    sp = az.ContinuousSelfPlay(GAMES, sampler, chunk=50, max_age=MAX_AGE, draw_threshold=DRAW_THRESHOLD)
    best_engine = 0.0
    while time.time() < deadline:
        t0 = time.time()
        key, k_play, k_eval = jax.random.split(key, 3)
        games, dropped = sp.collect(params, k_play, GAMES, SIMULATIONS, GAMMA)
        cur = [g for g in games if g[3] is not None]
        nor = [g for g in games if g[3] is None]
        curr_success = float(np.mean([g[4] == g[3] for g in cur])) if cur else 0.0
        obs = np.concatenate([g[0] for g in games]); pi = np.concatenate([g[1] for g in games]); z = np.concatenate([g[2] for g in games])
        replay.append((obs, pi, z))
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
        lens = [g[5] for g in nor]
        row = {"iteration": it, "avg_len": float(np.mean(lens)) if lens else 0.0,
               "first_win": float(np.mean([g[4] == 0 for g in nor])) if nor else 0.0,
               "second_win": float(np.mean([g[4] == 1 for g in nor])) if nor else 0.0, "draw": float(np.mean([g[4] < 0 for g in nor])) if nor else 0.0,
               "policy_loss": float(pl), "value_loss": float(vl), "curr_k": curr_k, "curr_success": curr_success,
               "curr_games": len(cur), "dropped": dropped, "gamma": GAMMA, "max_ply": int(max(lens)) if lens else 0}
        if curr_success >= CURR_PASS and curr_k < CURR_MAX:
            curr_k = min(CURR_MAX, curr_k + CURR_STEP)
            ck_state["k"] = curr_k
        if it % EVAL_EVERY == 0:
            w, d, l = (int(x) for x in az.evaluate(params, k_eval, batch=EVAL_GAMES, num_simulations=SIMULATIONS, max_len=MAX_LEN,
                                                   opponent="greedy", draw_threshold=DRAW_THRESHOLD))
            row["vs_greedy"], row["vs_greedy_score"] = [w, d, l], (w + 0.5 * d) / EVAL_GAMES
        if it % ENGINE_EVAL_EVERY == 0:
            w, d, l = eb.evaluate_vs_engine(params, engine, ENGINE_EVAL_GAMES, SIMULATIONS, MAX_LEN, DRAW_THRESHOLD, SEED + it)
            row["vs_engine"], row["vs_engine_score"] = [w, d, l], (w + 0.5 * d) / ENGINE_EVAL_GAMES
        row["sec"] = time.time() - t0
        history.append(row)
        msg = (f"[{it}] {row['sec']:.0f}s | 처음 배치 판 평균 {row['avg_len']:.0f}수, 선승 {row['first_win']:.0%} 후승 {row['second_win']:.0%} 무 {row['draw']:.0%} "
               f"(최장 {row['max_ply']}, 뺀 판 {dropped}) | 손실 정책 {pl:.3f} 가치 {vl:.3f} | 막판 {row['curr_k']}수 이내 {len(cur)}판 성공 {curr_success:.0%}")
        if "vs_greedy" in row:
            msg += f" | 전술 상대 {row['vs_greedy']} ({row['vs_greedy_score']:.2f})"
        if "vs_engine" in row:
            msg += f" | Rust 엔진 {row['vs_engine']} ({row['vs_engine_score']:.2f})"
        log(msg)
        az.save(str(ckpt), params, opt_state, it, history)
        hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=1))
        if "vs_engine" in row or it % EVAL_EVERY == 0:
            put_attr("progress", msg)
        sync()  # 매 반복 GCS 로 (Spot 회수 때 잃는 학습을 한 반복으로 줄임)
        if "vs_engine" in row:
            best_engine = max(best_engine, row["vs_engine_score"])
        if it - start_it >= STOP_AFTER and best_engine < STOP_SCORE:
            log(f"멈춤: {STOP_AFTER}반복 동안 엔진 상대 최고 {best_engine:.2f} < {STOP_SCORE}")
            break
    engine.close()
    lines = [l for l in (OUT / "log.txt").read_text().splitlines() if "엔진 상대" in l or "Rust 엔진" in l or "사전학습" in l]
    put_attr("summary", "\n".join(lines[-80:]))
    sync()


if __name__ == "__main__":
    main()

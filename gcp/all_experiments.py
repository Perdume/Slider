"""Slider: 가능한 개선을 전부 SPRT 로 검증 (Google Cloud VM 처럼 코어 많은 리눅스 머신용).

colab/slider_all_experiments.ipynb 와 같은 실험을 스크립트 하나로 돌린다.
  A. 탐색 개선 (pvs / lmr / nmp) vs 기본 탐색
  B. 수 제한 없는 자가대전으로 가중치 재튜닝 -> 새 가중치 vs 기준 가중치
  C. 같은 데이터로 특징 입력 신경망 -> 깊이 고정 / 시간 제한 대국

결과는 ~/slider_all/out/state.json 에 대국마다 저장된다. 중간에 끊기면 같은 명령을 다시 실행하면 끝난 것은 건너뛴다.
마지막에 요약을 출력하고 ~/slider_all/out/summary.txt 에 저장한다.
설정은 환경 변수로 바꿀 수 있다 (아래 SETTINGS 참고). 예: TUNE_GAMES=3000 python3 all_experiments.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import texel  # noqa: E402

REPO = HERE.parent


def env(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return type(default)(v) if not isinstance(default, bool) else v.lower() in ("1", "true", "yes")


SETTINGS = {
    "MATCH_TIME_MS": env("MATCH_TIME_MS", 30),      # 수당 시간 (ms)
    "SPRT_ELO0": env("SPRT_ELO0", 0.0),
    "SPRT_ELO1": env("SPRT_ELO1", 25.0),
    "SPRT_MAX_GAMES": env("SPRT_MAX_GAMES", 2000),  # 코어가 많으니 Colab 보다 넉넉하게
    "MAX_PLIES": env("MAX_PLIES", 1000),            # 수 제한 (사실상 없음)
    "SEED": env("SEED", 1),
    "SEARCH_TESTS": env("SEARCH_TESTS", "pvs,lmr,nmp"),
    "TUNE_GAMES": env("TUNE_GAMES", 2000),
    "TUNE_TIME_MS": env("TUNE_TIME_MS", 30),
    "DRAW_VALUE": env("DRAW_VALUE", 0.5),           # 0.0 = 무승부를 둘 다 진 것으로 (상수항만 바뀔 위험)
    "HIDDEN": env("HIDDEN", 32),
    "NN_EPOCHS": env("NN_EPOCHS", 60),
    "NN_DEPTH": env("NN_DEPTH", 4),
    "RUN_SEARCH": env("RUN_SEARCH", True),
    "RUN_TUNE": env("RUN_TUNE", True),
    "RUN_NN": env("RUN_NN", True),
    "RUN_EXTRA": env("RUN_EXTRA", True),
    "RUN_SMP": env("RUN_SMP", True),
    "RUN_BIG_NN": env("RUN_BIG_NN", True),
    "BIG_HIDDEN": env("BIG_HIDDEN", 128),
    "RUN_STACK": env("RUN_STACK", True),
    "RUN_SPSA": env("RUN_SPSA", True),
    "SPSA_ITERS": env("SPSA_ITERS", 200),
    "SPSA_GAMES": env("SPSA_GAMES", 64),
    "SPSA_LR": env("SPSA_LR", 1.5),
    # SPSA 대국만 수 제한을 둔다: 아주 긴 판 하나가 반복 전체를 붙잡지 않게 (무승부는 SPSA 계산에서 0 이라 영향이 작음)
    "SPSA_MAX_PLIES": env("SPSA_MAX_PLIES", 300),
    "RUN_SPSA_W": env("RUN_SPSA_W", True),
    "SPSA_W_ITERS": env("SPSA_W_ITERS", 300),
}
S = SETTINGS
THREADS = os.cpu_count() or 2
WORK = pathlib.Path(os.environ.get("SLIDER_WORK", pathlib.Path.home() / "slider_all"))
OUT = WORK / "out"
(OUT / "data").mkdir(parents=True, exist_ok=True)
STATE_PATH = OUT / "state.json"
state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def log(*args):
    print(*args, flush=True)


def run(binary, args, env_extra=None, cwd=None):
    cmd = [str(binary), *map(str, args)]
    log("$", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         cwd=cwd, env={**os.environ, **(env_extra or {})})
    lines = []
    for line in p.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        log("   ", line)
    if p.wait() != 0:
        raise RuntimeError(f"실행 실패 (exit {p.returncode}): {cmd[0]}")
    return lines


def save_state():
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


def fmt(r):
    if "aWins" not in r:
        return ""
    return (f"{r['aWins']}승 {r['draws']}무 {r['bWins']}패 ({r['games']}판), Elo {r['elo']:+.0f} "
            f"({r['eloLow']:+.0f} ~ {r['eloHigh']:+.0f}), SPRT {r.get('sprt', '-')}")


def sprt(key, a_args, b_args, limit=None, note="", max_games=None, game_threads=None, bin_dir=None):
    if key in state:
        log(f"[{key}] 이미 끝남: {fmt(state[key])}")
        return state[key]
    limit = limit or ["--time-ms", S["MATCH_TIME_MS"]]
    lines = run((bin_dir or BIN) / "matchup", [*a_args, *b_args, "--games", max_games or S["SPRT_MAX_GAMES"], *limit,
                                               "--max-plies", S["MAX_PLIES"], "--sprt-elo0", S["SPRT_ELO0"], "--sprt-elo1", S["SPRT_ELO1"],
                                               "--threads", game_threads or THREADS, "--seed", S["SEED"] + 1000])
    r = json.loads([l for l in lines if l.startswith("RESULT ")][-1][len("RESULT "):])
    r["note"] = note
    state[key] = r
    save_state()
    log(f"[{key}] {fmt(r)}")
    return r


# ---------------------------------------------------------------- 빌드
log("설정:", SETTINGS, "| threads:", THREADS, "| 결과:", OUT)
TARGET = WORK / f"target_{S['HIDDEN']}"
run("cargo", ["build", "--release", "--bin", "selfplay", "--bin", "trainnet", "--bin", "matchup"], cwd=REPO / "rust",
    env_extra={"SLIDER_HIDDEN": str(S["HIDDEN"]), "CARGO_TARGET_DIR": str(TARGET)})
BIN = TARGET / "release"
W = REPO / "weights" / "tuned.json"
log("기준 가중치:", W.read_text().strip())
t_start = time.time()

# ---------------------------------------------------------------- A. 탐색 개선
if S["RUN_SEARCH"]:
    tests = [t for t in S["SEARCH_TESTS"].split(",") if t]
    for o in tests:
        sprt(f"search_{o}", ["--a", W, "--opts-a", o], ["--b", W], note=f"탐색 {o} vs 기본")
    passed = [o for o in tests if state.get(f"search_{o}", {}).get("sprt") == "H1"]
    if len(passed) >= 2:
        combo = ",".join(passed)
        sprt(f"search_{combo}", ["--a", W, "--opts-a", combo], ["--b", W], note=f"탐색 {combo} vs 기본")
    log("통과한 탐색 개선:", passed or "없음")

# ---------------------------------------------------------------- B-1. 데이터
tune_data = OUT / "data" / f"tune_{S['TUNE_GAMES']}g_{S['TUNE_TIME_MS']}ms_p{S['MAX_PLIES']}.jsonl"
if (S["RUN_TUNE"] or S["RUN_NN"]) and not (tune_data.exists() and tune_data.stat().st_size > 0):
    t0 = time.time()
    partial = tune_data.with_suffix(".partial")
    run(BIN / "selfplay", ["--games", S["TUNE_GAMES"], "--time-ms", S["TUNE_TIME_MS"], "--random-plies", 4, "--epsilon", 0.05,
                           "--max-plies", S["MAX_PLIES"], "--threads", THREADS, "--seed", S["SEED"], "--weights", W,
                           "--positions", "yes", "--out", partial])
    partial.replace(tune_data)  # 끝까지 만든 파일만 이름을 바꿔 둔다 (끊기면 다시 만든다)
    log(f"자가대전 {time.time() - t0:.0f}초 -> {tune_data}")

# ---------------------------------------------------------------- B-2. 재튜닝
if S["RUN_TUNE"]:
    new_path = OUT / "data" / "weights_retuned.json"
    base = json.loads(W.read_text())
    if not new_path.exists() and "retune_skipped" not in state:
        X, y, gid, t = texel.load_samples([(1, tune_data)])
        y = np.where(y == 0.5, S["DRAW_VALUE"], y)
        w0 = texel.to_vector(base)
        texel.diagnose(X, y, t, w0)
        new = texel.tune(X, y, gid, w0, seed=S["SEED"])
        if new is None:
            state["retune_skipped"] = {"note": "K 를 정할 수 없어 튜닝 건너뜀"}
            save_state()
        else:
            texel.write_weights(new_path, new)
    if new_path.exists():
        new = json.loads(new_path.read_text())
        log(pd.DataFrame({"기준": base, "재튜닝": new}).assign(변화=lambda d: d["재튜닝"] - d["기준"]).to_string())
        state["retune_weights"] = {"note": json.dumps(new)}
        save_state()
        if new != base:
            sprt("retune", ["--a", new_path], ["--b", W], note="재튜닝 가중치 vs 기준 가중치")
        else:
            log("재튜닝 결과가 기준과 같아서 대국을 건너뜁니다")

# ---------------------------------------------------------------- C. 신경망
if S["RUN_NN"]:
    net = OUT / "data" / f"net_feat_{S['HIDDEN']}_{tune_data.stem}.txt"
    if not net.exists():
        lines = run(BIN / "trainnet", ["--data", tune_data, "--weights", W, "--out", net, "--epochs", S["NN_EPOCHS"],
                                       "--lambda", 1.0, "--features", "yes", "--seed", S["SEED"]])
        best = [l for l in lines if "가장 좋은 검증 손실" in l]
        state["nn_train"] = {"note": best[-1].strip() if best else ""}
        save_state()
    sprt("nn_depth", ["--a", W, "--net-a", net], ["--b", W], limit=["--time-ms", 100000, "--depth", S["NN_DEPTH"]],
         note=f"신경망, 깊이 {S['NN_DEPTH']} 고정")
    sprt("nn_time", ["--a", W, "--net-a", net], ["--b", W], note=f"신경망, 수당 {S['MATCH_TIME_MS']}ms")

# ---------------------------------------------------------------- D. 2차 실험
# 시간을 늘리면 결론이 바뀌는지: 시간 2배의 값(짧은/긴 시간), LMR 과 신경망을 긴 시간에서 다시, 신경망에 속도 손해만큼 시간 보정
if S["RUN_EXTRA"]:
    net = OUT / "data" / f"net_feat_{S['HIDDEN']}_{tune_data.stem}.txt"
    t = lambda ms: ["--time-ms", ms]
    sprt("time_60_vs_30", ["--a", W, "--time-ms-a", 60], ["--b", W, "--time-ms-b", 30], note="같은 엔진, 시간 2배 (60 vs 30ms)")
    sprt("time_300_vs_150", ["--a", W, "--time-ms-a", 300], ["--b", W, "--time-ms-b", 150], note="같은 엔진, 시간 2배 (300 vs 150ms)",
         max_games=600)
    sprt("lmr2_vs_lmr", ["--a", W, "--opts-a", "lmr2"], ["--b", W, "--opts-b", "lmr"], note="더 센 LMR vs LMR (30ms)")
    sprt("lmr_150ms", ["--a", W, "--opts-a", "lmr"], ["--b", W], limit=t(150), note="LMR vs 기본 (150ms)", max_games=800)
    if net.exists():
        sprt("nn_time_x1.6", ["--a", W, "--net-a", net, "--time-ms-a", 48], ["--b", W, "--time-ms-b", 30],
             note="신경망에 시간 1.6배 (48 vs 30ms, 속도 손해 보정)")
        sprt("nn_150ms", ["--a", W, "--net-a", net], ["--b", W], limit=t(150), note="신경망 vs 선형 (150ms)", max_games=800)
        sprt("nn_300ms", ["--a", W, "--net-a", net], ["--b", W], limit=t(300), note="신경망 vs 선형 (300ms)", max_games=400)

# ---------------------------------------------------------------- E. 병렬 탐색, 큰 신경망
if S["RUN_SMP"]:
    # 한 수를 스레드 4개로 탐색 (A) vs 1개 (B). 동시에 두는 판 수는 코어 / 4 로 줄여서 서로 코어를 뺏지 않게 한다
    gt = max(1, THREADS // 4)
    sprt("smp4_30ms", ["--a", W, "--search-threads-a", 4], ["--b", W], note="병렬 탐색 4스레드 vs 1스레드 (30ms)", game_threads=gt)
    sprt("smp4_150ms", ["--a", W, "--search-threads-a", 4], ["--b", W], limit=["--time-ms", 150],
         note="병렬 탐색 4스레드 vs 1스레드 (150ms)", game_threads=gt, max_games=300)
    # 16스레드: 한 판이 코어 16개를 쓰므로 동시에 두는 판 수는 코어 / 16
    gt16 = max(1, THREADS // 16)
    sprt("smp16_30ms", ["--a", W, "--search-threads-a", 16], ["--b", W], note="병렬 탐색 16스레드 vs 1스레드 (30ms)",
         game_threads=gt16, max_games=300)
    sprt("smp16_vs_4_30ms", ["--a", W, "--search-threads-a", 16], ["--b", W, "--search-threads-b", 4],
         note="병렬 탐색 16스레드 vs 4스레드 (30ms)", game_threads=gt16, max_games=300)
    sprt("smp16_60_vs_30", ["--a", W, "--search-threads-a", 16, "--time-ms-a", 60], ["--b", W, "--search-threads-b", 16, "--time-ms-b", 30],
         note="16스레드끼리 시간 2배 (60 vs 30ms)", game_threads=gt16, max_games=300)

if S["RUN_SMP"] or S["RUN_BIG_NN"]:
    # 섞어 쓰기: 1단계(안전한 수 찾기)는 선형 평가로 깊게, 2단계(안전한 수 중 고르기)만 신경망으로 깊이 4
    net32 = OUT / "data" / f"net_feat_{S['HIDDEN']}_{tune_data.stem}.txt"
    if net32.exists():
        sprt("nnplan_30ms", ["--a", W, "--net-a", net32, "--opts-a", "nnplan"], ["--b", W], note="신경망은 후보 고르기에만 (30ms)")
        sprt("nnplan_150ms", ["--a", W, "--net-a", net32, "--opts-a", "nnplan"], ["--b", W], limit=["--time-ms", 150],
             note="신경망은 후보 고르기에만 (150ms)", max_games=300)

if S["RUN_BIG_NN"]:
    h = S["BIG_HIDDEN"]
    big_target = WORK / f"target_{h}"
    run("cargo", ["build", "--release", "--bin", "trainnet", "--bin", "matchup"], cwd=REPO / "rust",
        env_extra={"SLIDER_HIDDEN": str(h), "CARGO_TARGET_DIR": str(big_target)})
    big_bin = big_target / "release"
    big_net = OUT / "data" / f"net_feat_{h}_{tune_data.stem}.txt"
    if not big_net.exists():
        lines = run(big_bin / "trainnet", ["--data", tune_data, "--weights", W, "--out", big_net, "--epochs", 30,
                                           "--lambda", 1.0, "--features", "yes", "--seed", S["SEED"]])
        best = [l for l in lines if "가장 좋은 검증 손실" in l]
        state[f"nn{h}_train"] = {"note": best[-1].strip() if best else ""}
        save_state()
    sprt(f"nn{h}_depth", ["--a", W, "--net-a", big_net], ["--b", W], limit=["--time-ms", 100000, "--depth", S["NN_DEPTH"]],
         note=f"신경망 은닉 {h}, 깊이 {S['NN_DEPTH']} 고정", bin_dir=big_bin)
    sprt(f"nn{h}_time", ["--a", W, "--net-a", big_net], ["--b", W], note=f"신경망 은닉 {h}, 수당 {S['MATCH_TIME_MS']}ms",
         bin_dir=big_bin)

# ---------------------------------------------------------------- F. 개선 합치기 확인 + 탐색 상수 자동 튜닝(SPSA)
if S["RUN_STACK"]:
    gt = max(1, THREADS // 4)
    sprt("stack_lmr2_smp4_30ms", ["--a", W, "--opts-a", "lmr2", "--search-threads-a", 4], ["--b", W],
         note="lmr2 + 4스레드 vs 기본 (30ms)", game_threads=gt, max_games=400)
    sprt("stack_lmr2_smp4_150ms", ["--a", W, "--opts-a", "lmr2", "--search-threads-a", 4], ["--b", W], limit=["--time-ms", 150],
         note="lmr2 + 4스레드 vs 기본 (150ms)", game_threads=gt, max_games=300)

SPSA_PAR = env("SPSA_PAR", 4)  # 동시에 돌리는 흔들기 수 (판 하나가 길면 다른 대국이 코어를 채운다)


def spsa_batch(k0, n, make_cmd):
    """반복 k0 .. k0+n-1 의 대국을 동시에 돌리고 (A쪽 이긴 판 - B쪽 이긴 판) / 판 수 목록을 돌려준다."""
    procs = [subprocess.Popen(list(map(str, make_cmd(k0 + i))), stdout=subprocess.PIPE, text=True) for i in range(n)]
    nets = []
    for p in procs:
        out, _ = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("matchup 실패")
        r = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])
        nets.append((r["aWins"] - r["bWins"]) / r["games"])
    return nets


# 이름, 시작값, 흔드는 폭 c, 최소, 최대 (기본값 = 예전 고정 상수)
SPSA_PARAMS = [("margin", 40, 8, 10, 120), ("trap", 60, 12, 0, 200), ("plan_ply", 5, 1, 3, 9),
               ("lmr_depth", 3, 1, 2, 6), ("lmr_index", 3, 1, 1, 8), ("lmr2_depth", 5, 1, 3, 8), ("lmr2_index", 6, 1, 3, 12)]


def tune_str(theta):
    return ",".join(f"{n}={int(round(min(hi, max(lo, v))))}" for (n, _, _, lo, hi), v in zip(SPSA_PARAMS, theta))


if S["RUN_SPSA"]:
    # SPSA: 상수를 한꺼번에 +c / -c 방향(무작위 부호)으로 흔든 두 엔진을 맞붙여, 이긴 쪽으로 조금씩 옮긴다 (Stockfish 가 상수를 맞추는 방식)
    sp = state.get("spsa", {"iter": 0, "theta": [float(p[1]) for p in SPSA_PARAMS], "log": []})
    c = np.array([p[2] for p in SPSA_PARAMS], float)
    lo = np.array([p[3] for p in SPSA_PARAMS], float)
    hi = np.array([p[4] for p in SPSA_PARAMS], float)
    while sp["iter"] < S["SPSA_ITERS"]:
        k0 = sp["iter"]
        n = min(SPSA_PAR, S["SPSA_ITERS"] - k0)
        theta = np.array(sp["theta"], float)
        deltas = [np.random.default_rng(S["SEED"] * 1_000_003 + k).choice([-1.0, 1.0], size=len(SPSA_PARAMS)) for k in range(k0, k0 + n)]

        def make_cmd(k, theta=theta, deltas=deltas, k0=k0):
            d = deltas[k - k0]
            plus, minus = np.clip(theta + c * d, lo, hi), np.clip(theta - c * d, lo, hi)
            return [BIN / "matchup", "--a", W, "--opts-a", "lmr2", "--tune-a", tune_str(plus), "--b", W, "--opts-b", "lmr2",
                    "--tune-b", tune_str(minus), "--games", S["SPSA_GAMES"], "--time-ms", S["MATCH_TIME_MS"],
                    "--max-plies", S["SPSA_MAX_PLIES"], "--threads", max(1, THREADS // n), "--seed", 50_000 + k]

        nets = spsa_batch(k0, n, make_cmd)
        for d, net in zip(deltas, nets):
            theta = np.clip(theta + S["SPSA_LR"] * c * net * d, lo, hi)
        sp["theta"], sp["iter"] = theta.tolist(), k0 + n
        sp["log"].append([k0 + n, [round(x, 3) for x in nets], tune_str(theta)])
        state["spsa"] = sp
        save_state()
        log(f"[SPSA {sp['iter']}/{S['SPSA_ITERS']}] {tune_str(theta)}")
    final = tune_str(sp["theta"])
    state["spsa_result"] = {"note": "SPSA 결과 " + final}
    save_state()
    log("SPSA 결과:", final)
    sprt("spsa_30ms", ["--a", W, "--opts-a", "lmr2", "--tune-a", final], ["--b", W, "--opts-b", "lmr2"],
         note="SPSA 상수 vs 기본 상수 (lmr2, 30ms)")
    sprt("spsa_150ms", ["--a", W, "--opts-a", "lmr2", "--tune-a", final], ["--b", W, "--opts-b", "lmr2"], limit=["--time-ms", 150],
         note="SPSA 상수 vs 기본 상수 (lmr2, 150ms)", max_games=400)

# ---------------------------------------------------------------- G. 평가 가중치 SPSA (대국을 직접 이기도록)
# 판 결과를 맞히는 튜닝(Texel)은 수 제한 없는 데이터에서 신호가 없었다 (상관 +0.01). 그래서 대국 결과로 직접 맞춘다.
W_NAMES = ["threat1", "threat2", "threat2PerThreat", "mobility", "reach", "plan2", "plan3", "plan4", "plan5", "position", "tempo"]


def write_w(path, vec):
    path.write_text(json.dumps({n: int(round(v)) for n, v in zip(W_NAMES, vec)}))
    return path


if S["RUN_SPSA_W"]:
    base_w = json.loads(W.read_text())
    w0 = np.array([float(base_w[n]) for n in W_NAMES])
    cw = np.maximum(4.0, 0.15 * np.abs(w0))
    lo_w, hi_w = w0 - np.maximum(200.0, 3 * np.abs(w0)), w0 + np.maximum(200.0, 3 * np.abs(w0))
    tune_final = state.get("spsa_result", {}).get("note", "").replace("SPSA 결과 ", "")
    spw = state.get("spsa_w", {"iter": 0, "theta": w0.tolist(), "log": []})
    pw, mw = WORK / "spsa_w_plus.json", WORK / "spsa_w_minus.json"
    while spw["iter"] < S["SPSA_W_ITERS"]:
        k0 = spw["iter"]
        n = min(SPSA_PAR, S["SPSA_W_ITERS"] - k0)
        theta = np.array(spw["theta"], float)
        deltas = [np.random.default_rng(S["SEED"] * 7_000_003 + k).choice([-1.0, 1.0], size=len(W_NAMES)) for k in range(k0, k0 + n)]
        for i, d in enumerate(deltas):
            write_w(WORK / f"spsa_w_plus{i}.json", np.clip(theta + cw * d, lo_w, hi_w))
            write_w(WORK / f"spsa_w_minus{i}.json", np.clip(theta - cw * d, lo_w, hi_w))

        def make_cmd(k, k0=k0, n=n):
            i = k - k0
            return [BIN / "matchup", "--a", WORK / f"spsa_w_plus{i}.json", "--opts-a", "lmr2", "--b", WORK / f"spsa_w_minus{i}.json",
                    "--opts-b", "lmr2", "--games", S["SPSA_GAMES"], "--time-ms", S["MATCH_TIME_MS"], "--max-plies", S["SPSA_MAX_PLIES"],
                    "--threads", max(1, THREADS // n), "--seed", 90_000 + k]

        nets = spsa_batch(k0, n, make_cmd)
        for d, net in zip(deltas, nets):
            theta = np.clip(theta + S["SPSA_LR"] * cw * net * d, lo_w, hi_w)
        spw["theta"], spw["iter"] = theta.tolist(), k0 + n
        spw["log"].append([k0 + n, [round(x, 3) for x in nets], [int(round(v)) for v in theta]])
        state["spsa_w"] = spw
        save_state()
        k = spw["iter"] - 1
        if True:
            log(f"[가중치 SPSA {k + 1}/{S['SPSA_W_ITERS']}] " + ", ".join(f"{n}={int(round(v))}" for n, v in zip(W_NAMES, theta)))
    new_w = write_w(OUT / "data" / "weights_spsa.json", np.array(spw["theta"]))
    state["spsa_w_result"] = {"note": "가중치 SPSA 결과 " + new_w.read_text()}
    save_state()
    extra = ["--tune-a", tune_final] if tune_final else []
    sprt("final_30ms", ["--a", new_w, "--opts-a", "lmr2", *extra], ["--b", W, "--opts-b", "lmr2"],
         note="새 가중치+상수 vs 지금 게임 설정 (lmr2, 30ms)")
    sprt("final_150ms", ["--a", new_w, "--opts-a", "lmr2", *extra], ["--b", W, "--opts-b", "lmr2"], limit=["--time-ms", 150],
         note="새 가중치+상수 vs 지금 게임 설정 (lmr2, 150ms)", max_games=400)

# ---------------------------------------------------------------- H. contempt (무승부를 수를 고르는 쪽에게만 나쁘게)
if env("RUN_CONTEMPT", True):
    for cval in (30, 80):
        sprt(f"contempt{cval}_30ms", ["--a", W, "--opts-a", "lmr2", "--tune-a", f"contempt={cval}"], ["--b", W, "--opts-b", "lmr2"],
             note=f"contempt {cval} vs 0 (lmr2, 30ms)", max_games=1000)
    # 형세에 따라: 이기는 중(+100 넘음)이면 무승부를 50 나쁘게, 지는 중(-100 아래)이면 50 좋게
    sprt("dyncontempt50_30ms", ["--a", W, "--opts-a", "lmr2", "--tune-a", "dyn_contempt=50,dyn_threshold=100"],
         ["--b", W, "--opts-b", "lmr2"], note="형세 contempt 50 vs 0 (lmr2, 30ms)", max_games=1000)

# ---------------------------------------------------------------- 요약
lines = [f"설정: {SETTINGS} | threads {THREADS} | 이번 실행 {time.time() - t_start:.0f}초"]
for key, r in state.items():
    if key == "spsa":
        lines.append(f"{key:<22} 반복 {r['iter']}회, 현재 {tune_str(r['theta'])}")
        continue
    if key == "spsa_w":
        lines.append(f"{key:<22} 반복 {r['iter']}회, 현재 " + ", ".join(f"{n}={int(round(v))}" for n, v in zip(W_NAMES, r["theta"])))
        continue
    lines.append(f"{key:<22} {r.get('note', '')[:60]:<40} {fmt(r)}")
text = "\n".join(lines)
(OUT / "summary.txt").write_text(text + "\n\nJSON:\n" + json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
log("\n================ 요약 ================")
log(text)
log("\n저장:", OUT / "summary.txt")

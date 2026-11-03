"""새 규칙(cycle)에서 탐색 상수 + 평가 가중치를 같이 SPSA 로 다시 맞춘다 (VM 용).

시작점 = 지금 게임 설정 (Tunables::game + weights/tuned.json). 끝나면 지금 게임 설정과 SPRT 로 비교한다.
진행 상황: ~/slider_spsa/state.json (끊겨도 이어서), ~/slider_spsa/progress.txt (밖으로 내보내는 요약)
"""
import json
import os
import pathlib
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path.home() / "slider_spsa"
WORK.mkdir(parents=True, exist_ok=True)
STATE = WORK / "state.json"
PROGRESS = WORK / "progress.txt"
BIN = REPO / "rust" / "target" / "release"
W = REPO / "weights" / "tuned.json"
THREADS = os.cpu_count() or 2
ITERS = int(os.environ.get("SPSA_ITERS", 320))
GAMES = int(os.environ.get("SPSA_GAMES", 64))
PAR = int(os.environ.get("SPSA_PAR", 4))
LR = float(os.environ.get("SPSA_LR", 1.5))
TIME_MS = 30
MAX_PLIES = 400
SEED = 7

# 탐색 상수: 이름, 시작값(game), 흔드는 폭, 최소, 최대
T_PARAMS = [("margin", 39, 8, 10, 120), ("trap", 38, 12, 0, 200), ("plan_ply", 9, 1, 3, 13),
            ("lmr_depth", 2, 1, 1, 6), ("lmr_index", 1, 1, 1, 8), ("lmr2_depth", 3, 1, 2, 8), ("lmr2_index", 6, 1, 2, 12)]
W_NAMES = ["threat1", "threat2", "threat2PerThreat", "mobility", "reach", "plan2", "plan3", "plan4", "plan5", "position", "tempo"]
w0 = json.loads(W.read_text())
W_PARAMS = [(n, float(w0[n]), max(4.0, 0.15 * abs(w0[n])), w0[n] - max(200.0, 3 * abs(w0[n])), w0[n] + max(200.0, 3 * abs(w0[n])))
            for n in W_NAMES]
PARAMS = T_PARAMS + W_PARAMS
NT = len(T_PARAMS)
C = np.array([p[2] for p in PARAMS], float)
LO = np.array([p[3] for p in PARAMS], float)
HI = np.array([p[4] for p in PARAMS], float)


def log(msg):
    print(msg, flush=True)


def tune_str(v):
    return ",".join(f"{n}={int(round(x))}" for (n, *_), x in zip(T_PARAMS, v[:NT]))


def write_w(path, v):
    path.write_text(json.dumps({n: int(round(x)) for n, x in zip(W_NAMES, v[NT:])}))
    return path


def matchup(args):
    p = subprocess.run([str(BIN / "matchup"), *map(str, args)], capture_output=True, text=True)
    res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not res:
        raise RuntimeError(f"matchup 실패: {p.stderr[-300:]}")
    return json.loads(res[-1][len("RESULT "):])


def main():
    subprocess.run(["cargo", "build", "--release", "-q", "--bin", "matchup"], cwd=REPO / "rust", check=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {"iter": 0, "theta": [p[1] for p in PARAMS], "log": []}
    while st["iter"] < ITERS:
        k0 = st["iter"]
        n = min(PAR, ITERS - k0)
        theta = np.array(st["theta"], float)
        deltas = [np.random.default_rng(SEED * 1_000_003 + k).choice([-1.0, 1.0], size=len(PARAMS)) for k in range(k0, k0 + n)]
        procs = []
        for i, d in enumerate(deltas):
            plus, minus = np.clip(theta + C * d, LO, HI), np.clip(theta - C * d, LO, HI)
            pw, mw = write_w(WORK / f"plus{i}.json", plus), write_w(WORK / f"minus{i}.json", minus)
            cmd = [BIN / "matchup", "--rules", "cycle", "--a", pw, "--opts-a", "lmr2", "--tune-a", tune_str(plus),
                   "--b", mw, "--opts-b", "lmr2", "--tune-b", tune_str(minus), "--games", GAMES, "--time-ms", TIME_MS,
                   "--max-plies", MAX_PLIES, "--threads", max(1, THREADS // n), "--seed", 50_000 + k0 + i]
            procs.append(subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, text=True))
        nets = []
        for p in procs:
            out, _ = p.communicate()
            r = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])
            nets.append((r["aWins"] - r["bWins"]) / r["games"])
        for d, net in zip(deltas, nets):
            theta = np.clip(theta + LR * C * net * d, LO, HI)
        st["theta"], st["iter"] = theta.tolist(), k0 + n
        st["log"].append([k0 + n, [round(x, 3) for x in nets]])
        STATE.write_text(json.dumps(st))
        line = f"[SPSA {st['iter']}/{ITERS}] {tune_str(theta)} | " + ", ".join(f"{n}={int(round(x))}" for n, x in zip(W_NAMES, theta[NT:]))
        log(line)
        PROGRESS.write_text(line + "\n" + "\n".join(st.get("results", [])))

    theta = np.array(st["theta"], float)
    new_w = write_w(WORK / "weights_cycle.json", theta)
    tune = tune_str(theta)
    st.setdefault("results", [])
    for key, ms, mg in (("final_30ms", 30, 2000), ("final_150ms", 150, 600)):
        if any(r.startswith(key) for r in st["results"]):
            continue
        r = matchup(["--rules", "cycle", "--a", new_w, "--opts-a", "lmr2", "--tune-a", tune, "--b", W, "--opts-b", "lmr2",
                     "--tune-b", "game", "--games", mg, "--time-ms", ms, "--max-plies", 1000, "--sprt-elo0", 0,
                     "--sprt-elo1", 25, "--threads", THREADS, "--seed", 1001])
        st["results"].append(f"{key} 새 상수+가중치 vs 지금 게임 설정: {r['aWins']}승 {r['draws']}무 {r['bWins']}패 ({r['games']}판) "
                             f"Elo {r['elo']:+.0f} ({r['eloLow']:+.0f} ~ {r['eloHigh']:+.0f}) {r.get('sprt')}")
        STATE.write_text(json.dumps(st))
        log(st["results"][-1])
    summary = f"tune: {tune}\nweights: {new_w.read_text()}\n" + "\n".join(st["results"])
    PROGRESS.write_text(summary)
    (WORK / "summary.txt").write_text(summary)
    log(summary)


if __name__ == "__main__":
    main()

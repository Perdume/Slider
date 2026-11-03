"""VM 용 SPRT 대기열: gcp/queue.json 에 적힌 대국을 차례로 돌린다 (끝난 것은 건너뜀).

queue.json 항목: {"key": 이름, "note": 설명, "a": [A쪽 matchup 인자], "b": [B쪽 인자],
                  "time_ms": 30, "max_games": 2000, "search_threads": 1}
인자 안의 "{W}" 는 기준 가중치 파일 경로로 바뀐다.
결과: ~/slider_queue/state.json (대국마다 저장), ~/slider_queue/summary.txt
"""
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path.home() / "slider_queue"
WORK.mkdir(parents=True, exist_ok=True)
STATE = WORK / "state.json"
THREADS = os.cpu_count() or 2
W = str(REPO / "weights" / "tuned.json")


def log(*a):
    print(*a, flush=True)


def main():
    queue = json.loads((REPO / "gcp" / "queue.json").read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    subprocess.run(["cargo", "build", "--release", "-q", "--bin", "matchup"], cwd=REPO / "rust", check=True)
    matchup = REPO / "rust" / "target" / "release" / "matchup"
    for q in queue:
        key = q["key"]
        if key in state:
            log(f"[{key}] 이미 끝남")
            continue
        st = int(q.get("search_threads", 1))
        args = [str(a).replace("{W}", W) for a in q["a"] + q["b"]]
        cmd = [str(matchup), *args, "--games", str(q.get("max_games", 2000)), "--time-ms", str(q.get("time_ms", 30)),
               "--max-plies", "1000", "--sprt-elo0", "0", "--sprt-elo1", str(q.get("elo1", 25)),
               "--threads", str(max(1, THREADS // st)), "--seed", "1001"]
        log("$", " ".join(cmd))
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        r = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][len("RESULT "):])
        r["note"] = q.get("note", "")
        state[key] = r
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        log(f"[{key}] {r['aWins']}승 {r['draws']}무 {r['bWins']}패 ({r['games']}판), Elo {r['elo']:+.0f} "
            f"({r['eloLow']:+.0f} ~ {r['eloHigh']:+.0f}), SPRT {r.get('sprt')} | {r['note']}")
    lines = []
    for k, r in state.items():
        lines.append(f"{k:<20} {r['note']:<45} {r['aWins']}승 {r['draws']}무 {r['bWins']}패 ({r['games']}판), "
                     f"Elo {r['elo']:+.0f} ({r['eloLow']:+.0f} ~ {r['eloHigh']:+.0f}), SPRT {r.get('sprt')}")
    (WORK / "summary.txt").write_text("\n".join(lines) + "\n")
    log("\n================ 요약 ================\n" + "\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())

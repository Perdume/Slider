#!/usr/bin/env bash
# Google Cloud VM (Debian/Ubuntu) 에서 전체 실험을 돌린다. 끊기면 같은 명령을 다시 실행하면 이어서 한다.
#   curl -sSL https://raw.githubusercontent.com/Perdume/Slider/rust-port/gcp/run_all.sh | bash
# 결과: ~/slider_all/out/summary.txt (로그: ~/slider_all/run.log)
# 설정은 환경 변수로: 예) TUNE_GAMES=3000 SPRT_MAX_GAMES=3000 bash run_all.sh
# AUTO_SHUTDOWN=1 이면 끝난 뒤 VM 을 끈다 (크레딧 절약. 디스크 요금은 조금 남으니 다 쓰면 VM 을 삭제하세요)
set -euo pipefail

BRANCH="${BRANCH:-rust-port}"
REPO="$HOME/Slider"
mkdir -p "$HOME/slider_all"

if ! command -v cc >/dev/null || ! command -v git >/dev/null || ! python3 -c "import venv" 2>/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git curl python3-venv tmux
fi
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" reset -q --hard "origin/$BRANCH"
else
  git clone -q --depth 1 --branch "$BRANCH" https://github.com/Perdume/Slider.git "$REPO"
fi
git -C "$REPO" log --oneline -1

if [ ! -x "$HOME/venv/bin/python" ]; then
  python3 -m venv "$HOME/venv"
fi
"$HOME/venv/bin/pip" install -q numpy pandas scipy

echo "코어: $(nproc)개. 로그는 ~/slider_all/run.log"
"$HOME/venv/bin/python" -u "$REPO/gcp/all_experiments.py" 2>&1 | tee -a "$HOME/slider_all/run.log"

if [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
  echo "끝났습니다. 1분 뒤 VM 을 끕니다."
  sudo shutdown -h +1
fi

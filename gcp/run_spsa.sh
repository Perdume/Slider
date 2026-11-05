#!/usr/bin/env bash
# VM 에서 gcp/queue.json 의 SPSA (spsa_cycle.py) 를 돌린다. 끊기면 다시 실행하면 끝난 것은 건너뛴다.
set -euo pipefail
BRANCH="${BRANCH:-rust-port}"
REPO="$HOME/Slider"
if ! command -v cc >/dev/null || ! command -v git >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git curl python3
fi
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
python3 -c "import numpy" 2>/dev/null || { sudo apt-get update -qq; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-numpy; }
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" reset -q --hard "origin/$BRANCH"
else
  git clone -q --depth 1 --branch "$BRANCH" https://github.com/Perdume/Slider.git "$REPO"
fi
git -C "$REPO" log --oneline -1
python3 -u "$REPO/gcp/spsa_cycle.py"

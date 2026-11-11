#!/usr/bin/env bash
# GPU VM 에서 강화학습 (gcp/rl_vm.py). Deep Learning VM (Ubuntu 24.04, NVIDIA 드라이버 포함) 기준.
set -euo pipefail
BRANCH="${BRANCH:-rust-port}"
REPO="$HOME/Slider"
export DEBIAN_FRONTEND=noninteractive
if ! command -v cc >/dev/null || ! command -v git >/dev/null; then
  apt-get update -qq && apt-get install -y -qq build-essential git curl
fi
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" reset -q --hard "origin/$BRANCH"
else
  git clone -q --depth 1 --branch "$BRANCH" https://github.com/Perdume/Slider.git "$REPO"
fi
git -C "$REPO" log --oneline -1
(cd "$REPO/rust" && cargo build --release -q --bin selfplay --bin bestmove)
if [ ! -x "$HOME/venv/bin/python" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv -q --python 3.12 "$HOME/venv"
  uv pip install -q --python "$HOME/venv/bin/python" "jax[cuda12]" mctx optax numpy
fi
"$HOME/venv/bin/python" -c "import jax; print(jax.devices())"
"$HOME/venv/bin/python" -u "$REPO/gcp/rl_vm.py"

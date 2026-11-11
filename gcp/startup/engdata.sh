#!/bin/bash
# 새 규칙 엔진 판 2000판 (1000판 x seed 2개) 을 만들어 GCS 에 올리고 VM 을 끈다.
export HOME=/root
cd /root
put_attr() {
  curl -s -X PUT --data-binary @- -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/slider/$1" >/dev/null || true
}
echo "running $(date -u +%FT%TZ)" | put_attr status
{
  set -e
  if ! command -v cc >/dev/null || ! command -v git >/dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git curl
  fi
  [ -x "$HOME/.cargo/bin/cargo" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  rm -rf /root/Slider && git clone -q --depth 1 -b rust-port https://github.com/Perdume/Slider.git /root/Slider
  cd /root/Slider/rust && cargo build --release -q --bin selfplay
  for seed in 1 2; do
    f=/root/engine_cycle_1000g_30ms_s$seed.jsonl
    ./target/release/selfplay --rules cycle --games 1000 --time-ms 30 --opts lmr2 --weights /root/Slider/weights/tuned.json \
      --max-plies 1000 --random-plies 4 --epsilon 0.05 --threads $(nproc) --seed $seed --positions yes --out $f
    echo "seed $seed done $(date -u +%T)" | put_attr progress
    gsutil -q cp $f gs://slider-rl-eminent-glider/
  done
} > /root/engdata.log 2>&1 && echo "done $(date -u +%FT%TZ)" | put_attr status || { echo "failed $(date -u +%FT%TZ)" | put_attr status; tail -n 40 /root/engdata.log | put_attr error; }
sleep 10
shutdown -h now

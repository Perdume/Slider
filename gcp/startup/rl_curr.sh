#!/bin/bash
# Slider GPU 강화학습 (할인 + 역방향 커리큘럼 + 수 제한 1000). 끝나면(성공/실패) 결과를 guest attribute 와 GCS 에 남기고 VM 을 끈다.
export HOME=/root BUCKET=slider-rl-eminent-glider RL_HOURS=10 RL_UNTIL=1793807520 SIMULATIONS=64 GAMES=256 GAMMA=0.995 MAX_LEN=1000 CURR_FRAC=0.5
cd /root
put_attr() {
  curl -s -X PUT --data-binary @- -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/slider/$1" >/dev/null || true
}
echo "running $(date -u +%FT%TZ)" | put_attr status
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi -L 2>&1 | put_attr gpu
curl -sSL -o /root/run_rl.sh https://raw.githubusercontent.com/Perdume/Slider/rust-port/gcp/run_rl.sh
if bash /root/run_rl.sh > /root/rl.log 2>&1; then
  echo "done $(date -u +%FT%TZ)" | put_attr status
else
  echo "failed $(date -u +%FT%TZ)" | put_attr status
  tail -n 60 /root/rl.log | put_attr error
fi
gsutil -q cp /root/rl.log gs://$BUCKET/ || true
sleep 20
shutdown -h now

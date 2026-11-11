#!/bin/bash
# GPU 강화학습 VM 이 없으면(끝났거나 Spot 회수) GCS 저장 파일에서 이어받는 VM 을 띄운다. END_AT(UTC) 까지만.
# 환경변수: END_AT (예 2026-11-05T01:00:00Z), STARTUP (startup/ 안의 시작 스크립트 이름, 기본 rl_curr.sh)
G="${GCLOUD:-gcloud}"; [ -n "${CLOUDSDK_AUTH_ACCESS_TOKEN:-}" ] && G="env -u CLOUDSDK_AUTH_ACCESS_TOKEN $G"; PR="--project eminent-glider-465204-b2"; HERE="$(cd "$(dirname "$0")" && pwd)"; SG="${SG:-$HERE/startup}"; LEDGER="${LEDGER:-$HOME/slider_cost_ledger.txt}"
END=$(date -u -d "${END_AT:-2026-11-04T16:01:00Z}" +%s); NOW=$(date -u +%s)
line=$($G compute instances list $PR --filter="name=slider-gpu" --format="value(zone.basename(),status)")
zone=${line%%[[:space:]]*}; st=${line##*[[:space:]]}
if [ "$st" = RUNNING ] || [ "$st" = STAGING ] || [ "$st" = PROVISIONING ]; then echo "GPU VM 실행 중 ($zone)"; exit 0; fi
[ -n "$line" ] && { echo "남은 GPU VM 정리 ($st)"; $G compute instances delete slider-gpu --zone $zone $PR --quiet 2>&1 | tail -1; }
left=$(( (END - NOW) / 60 ))
if [ $left -lt 40 ]; then echo "끝낼 시각까지 ${left}분뿐이라 새로 띄우지 않음"; exit 0; fi
for z in us-east1-c us-east1-d us-west1-b us-west1-a us-central1-f us-central1-c us-central1-a us-central1-b; do
  if $G compute instances create slider-gpu $PR --zone $z --machine-type n1-standard-8 --accelerator type=nvidia-tesla-t4,count=1 \
      --maintenance-policy TERMINATE --provisioning-model SPOT --instance-termination-action DELETE --max-run-duration ${left}m \
      --image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release --boot-disk-size 50GB \
      --scopes cloud-platform --metadata enable-guest-attributes=TRUE --metadata-from-file startup-script=$SG/${STARTUP:-rl_curr.sh} > /tmp/gpu_create.txt 2>&1; then
    echo "이어받는 GPU VM 시작: $z, 최대 ${left}분"; echo "GPU cont VM start $(date -u +%H:%M) zone $z cap ${left}m" >> $LEDGER; exit 0
  fi
  echo "$z 실패: $(grep -iE 'quota|resources|error' /tmp/gpu_create.txt | head -1)"
done
exit 1

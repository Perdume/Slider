#!/bin/bash
# Slider SPRT 대기열: 부팅할 때마다 실행 (끝난 대국은 건너뜀). 끝나면(성공/실패) 결과를 guest attribute 로 남기고 VM 을 끈다.
export HOME=/root
cd /root
put_attr() {
  curl -s -X PUT --data-binary @- -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/slider/$1" >/dev/null || true
}
echo "running $(date -u +%FT%TZ)" | put_attr status
# 끝난 실험 결과를 1분마다 밖으로 내보낸다 (VM 이 중간에 지워져도 남게)
( while true; do
    python3 -c "
import json,pathlib
p=pathlib.Path('/root/slider_queue/state.json')
s=json.loads(p.read_text()) if p.exists() else {}
for k,r in s.items(): print(f\"{k:<16} {r['aWins']}승 {r['draws']}무 {r['bWins']}패 ({r['games']}판) Elo {r['elo']:+.0f} ({r['eloLow']:+.0f} ~ {r['eloHigh']:+.0f}) {r.get('sprt')}\")
" 2>/dev/null | put_attr partial
    sleep 60
  done ) &
curl -sSL -o /root/run_queue.sh https://raw.githubusercontent.com/Perdume/Slider/rust-port/gcp/run_queue.sh
if bash /root/run_queue.sh > /root/queue.log 2>&1; then
  echo "done $(date -u +%FT%TZ)" | put_attr status
else
  echo "failed $(date -u +%FT%TZ)" | put_attr status
  tail -n 60 /root/queue.log | put_attr error
fi
head -c 200000 /root/slider_queue/summary.txt 2>/dev/null | put_attr summary
tail -n 200 /root/queue.log | grep -E "^\[" | put_attr results
sleep 20
shutdown -h now

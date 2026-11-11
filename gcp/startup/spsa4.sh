#!/bin/bash
# Slider SPSA (새 규칙). 진행 요약을 1분마다 guest attribute 로 내보내고, 끝나면 VM 을 끈다.
export HOME=/root SPSA_OPTS="lmr2,hm" SPSA_EXTRA="nt_me:688:40:0:1500,nt_opp:-324:40:-1000:200,pos_nx:0:6:-100:100,mob_oc:0:4:-80:80" SPSA_ITERS=320
cd /root
put_attr() {
  curl -s -X PUT --data-binary @- -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/slider/$1" >/dev/null || true
}
echo "running $(date -u +%FT%TZ)" | put_attr status
( while true; do cat /root/slider_spsa/progress.txt 2>/dev/null | put_attr partial; sleep 60; done ) &
curl -sSL -o /root/run_spsa.sh https://raw.githubusercontent.com/Perdume/Slider/rust-port/gcp/run_spsa.sh
if bash /root/run_spsa.sh > /root/spsa.log 2>&1; then
  echo "done $(date -u +%FT%TZ)" | put_attr status
else
  echo "failed $(date -u +%FT%TZ)" | put_attr status
  tail -n 60 /root/spsa.log | put_attr error
fi
cat /root/slider_spsa/summary.txt 2>/dev/null | put_attr summary
cat /root/slider_spsa/progress.txt 2>/dev/null | put_attr partial
sleep 20
shutdown -h now

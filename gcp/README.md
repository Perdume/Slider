# Google Cloud 에서 전체 실험 돌리기

`colab/slider_all_experiments.ipynb` 와 같은 실험(탐색 개선 SPRT, 가중치 재튜닝, 신경망 재확인)을 코어 많은 VM 에서 돌립니다.
코어 수에 거의 비례해서 빨라지고, Colab(코어 2개)이 3~4시간 걸리는 양을 vCPU 32개 VM 은 1시간 안팎에 끝냅니다 (어림).
기본 설정은 Colab 보다 크게 잡았습니다 (자가대전 2000판, SPRT 최대 2000판).

## 1. VM 만들기 (콘솔)
Compute Engine -> VM 인스턴스 -> 인스턴스 만들기
- 리전: `us-central1` (아무 곳이나 됨. 스팟 가격은 리전마다 다름)
- 머신: **C2D -> `c2d-highcpu-32`** (없거나 할당량 오류가 나면 `c2d-highcpu-16`, `e2-highcpu-16` 등 더 작은 것)
- VM 프로비저닝 모델: **스팟** (훨씬 쌈. 구글이 회수할 수 있지만 다시 실행하면 이어서 함)
- 부팅 디스크: Debian 12, 20GB
- 만들기 -> 목록에서 **SSH** 버튼

또는 Cloud Shell 에서:
```bash
gcloud compute instances create slider-run --zone=us-central1-a --machine-type=c2d-highcpu-32 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=20GB
gcloud compute ssh slider-run --zone=us-central1-a
```

## 2. 실행 (VM 의 SSH 창에서)
```bash
tmux new -s slider        # SSH 가 끊겨도 계속 돌게 (다시 붙기: tmux attach -t slider)
curl -sSL https://raw.githubusercontent.com/Perdume/Slider/rust-port/gcp/run_all.sh | AUTO_SHUTDOWN=1 bash
```
- 중간에 VM 이 회수되거나 끊기면 VM 을 다시 시작하고 같은 명령을 실행하면 끝난 부분은 건너뜁니다.
- 끝나면 마지막에 요약이 출력되고 `~/slider_all/out/summary.txt` 에 저장됩니다. 요약을 복사해 오거나,
  SSH 창 오른쪽 위 톱니 -> 파일 다운로드로 `slider_all/out/summary.txt` 를 받으세요.

## 3. 끝나면 VM 삭제
`AUTO_SHUTDOWN=1` 은 VM 을 끄기만 합니다. 꺼진 VM 도 디스크 요금이 조금 나가니, 결과를 받은 뒤 VM 을 **삭제**하세요.

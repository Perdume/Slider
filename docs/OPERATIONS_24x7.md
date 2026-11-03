# Slider AI 24시간 운영 문서

Slider AI 학습을 사람이 안 보는 동안에도 계속 돌리기 위한 문서입니다. 이 레포를 처음 보는 사람이나 새 Claude 세션도
이 문서만 읽고 상태를 확인하고, 이어서 돌리고, 멈출 수 있게 썼습니다.

마지막 갱신: 2026-11-03 (새 규칙 = 대각선 배치, 말 하나씩 시계 방향)

---

## 1. 한눈에 보기

| 무엇 | 어디서 | 상태를 보는 곳 | 결과가 남는 곳 |
|---|---|---|---|
| 탐색 실험 (SPRT 대국) | GCP CPU VM `slider` (c2d-highcpu-32, Spot) | guest attribute `slider/partial` | guest attribute, Drive `slider_results` |
| 가중치·상수 튜닝 (SPSA) | GCP CPU VM `slider` | guest attribute `slider/partial` | guest attribute `slider/summary` → 레포 `Tunables::game` / `weights/` |
| 엔진 판 만들기 | GCP CPU VM `slider-data` | guest attribute `slider/progress` | GCS `gs://slider-rl-eminent-glider/engine_cycle_*.jsonl` |
| 강화학습 (합성곱 신경망) | GCP GPU VM `slider-gpu` (n1-standard-8 + T4, Spot) | guest attribute `slider/progress` | GCS `ckpt_cycle_conv_*.pkl`, `history_*.json`, `log.txt` → Drive |
| 강화학습 (무료) | Colab `colab/slider_alphazero_conv.ipynb` | 노트북 출력 | Drive `slider_results/alphazero/` |

- GCP 프로젝트: `eminent-glider-465204-b2` (결제 계정 `01D536-909123-E1B4FD`, 원화)
- 지역: `us-central1` (GPU 는 `us-central1-a` 에서 잘 잡힘. 안 되면 b → f → c)
- 할당량: CPU 전체 64개, GPU 1개 (둘 다 올려 둠)
- 버킷: `gs://slider-rl-eminent-glider` (us-central1)

## 2. 안전장치 (요금 폭탄 방지)

1. **모든 VM 에 최대 실행 시간 + 자동 삭제**: `--max-run-duration N --instance-termination-action DELETE`.
   제가 놓치거나 세션이 끊겨도 VM 은 그 시간에 디스크째 지워집니다. 실행 중에는 이 시간을 바꿀 수 없습니다
   (멈춘 VM 에서만 `set-scheduling --max-run-duration` 가능).
2. **일이 끝나면 VM 이 스스로 꺼짐**: 시작 스크립트가 마지막에 `shutdown -h now`. 꺼진 VM 은 시간 요금이 없고,
   남은 디스크(20~50GB, 한 달 내내 둬도 1~3달러)는 결과를 읽은 뒤 지웁니다.
3. **중간 결과를 계속 밖으로**: VM 은 1분마다(또는 실험 하나 끝날 때마다) guest attribute 에, 강화학습은 반복 10번마다 GCS 에
   저장합니다. VM 이 지워져도 거기까지는 남습니다.
4. **예산 알림**: 결제 계정에 26,000원 예산 (크레딧을 빼기 전 사용액 기준, 50/80/100% 에 메일). 알림일 뿐 자동으로 막지는 않습니다.
5. **비용 장부**: VM 을 띄우고 지울 때마다 시각을 적어서 직접 계산합니다 (콘솔 청구액은 몇 시간 늦게 반영됨). 아래 7절.
6. **총 한도**: 약 $18.5 (처음 크레딧 $10 + 추가 ₩13,500, 여유분 뺀 값). 넘기 전에 전부 정리합니다 (6절의 긴급 정지).

## 3. 스크립트

| 파일 | 하는 일 |
|---|---|
| `gcp/startup/queue.sh` | VM 시작 스크립트: `gcp/queue.json` 의 SPRT 실험을 차례로. 끝난 것은 건너뜀 |
| `gcp/startup/spsa.sh` | VM 시작 스크립트: `gcp/spsa_cycle.py` (새 규칙 SPSA, 탐색 상수 7 + 가중치 11) 후 지금 설정과 SPRT |
| `gcp/startup/engdata.sh` | VM 시작 스크립트: 새 규칙 엔진 판 1000판 x 2 를 만들어 GCS 에 올림 |
| `gcp/startup/rl.sh` | GPU VM 시작 스크립트: `gcp/rl_vm.py` (엔진 판 받기 → 사전학습 → 강화학습 RL_HOURS 시간) |
| `gcp/startup/rl_cont.sh` | 위와 같지만 GCS 저장 파일에서 이어서, `RL_UNTIL` 시각까지 |
| `gcp/relaunch_rl.sh` | GPU VM 이 없으면(끝났거나 Spot 회수) `rl_cont.sh` 로 이어받는 VM 을 띄움. 끝낼 시각 40분 전부터는 안 띄움 |
| `gcp/run_queue.sh`, `gcp/run_spsa.sh`, `gcp/run_rl.sh` | 시작 스크립트가 받아서 실행하는 본체 (레포에서 최신을 받음) |

시작 스크립트는 VM 안에서 `https://raw.githubusercontent.com/Perdume/Slider/rust-port/gcp/...` 를 받으므로,
**코드를 바꾸면 `rust-port` 에 푸시한 뒤에 VM 을 띄워야** 합니다.

## 4. 야간 자동 점검 (예약 작업)

이 세션에 예약해 둔 점검 (한국 시각). 각 점검은 `relaunch_rl.sh` 로 GPU VM 이 살아 있는지 보고, 없으면 이어받는 VM 을 띄우고,
진행 상황과 비용을 확인해서 보고합니다.

| 시각 (KST) | 할 일 |
|---|---|
| 11/03 23:30 | 점검 1: GPU 상태, SPSA 결과가 나왔으면 보고 + Drive 저장 + VM 삭제 + 통과 시 게임 설정 반영 |
| 11/04 02:20 | 점검 2: 첫 GPU VM(5시간) 끝 → 이어받는 VM 시작 (09:40 까지), 학습 기록 Drive 저장 |
| 11/04 05:30 | 점검 3: Spot 회수됐으면 이어받기 |
| 11/04 08:00 | 점검 4: 같음 |
| 11/04 10:00 | 최종 정리: GCS → Drive, 모든 VM/디스크 삭제 확인, 학습 곡선 요약, 비용 정리 |

예약 문구에 "$9 넘으면 정리" 가 남아 있지만, 한도는 약 $18.5 로 바뀌었습니다 (예약이 PC 에 묶여 있어 문구 수정에 승인이 필요해서
그대로 둠). **비용 장부의 한도를 기준으로 판단합니다.**

## 5. 자주 쓰는 명령

이 작업 환경에서는 `CLOUDSDK_AUTH_ACCESS_TOKEN` 환경 변수가 로그인을 덮어쓰므로 항상 이렇게 부릅니다.

```bash
G="env -u CLOUDSDK_AUTH_ACCESS_TOKEN gcloud"; PR="--project eminent-glider-465204-b2"

# 무엇이 돌고 있나 (비어 있으면 요금 나가는 VM/디스크 없음)
$G compute instances list $PR
$G compute disks list $PR

# 진행 상황 (VM 이름과 zone 을 맞춰서)
$G compute instances get-guest-attributes slider-gpu --zone us-central1-a $PR --query-path=slider/ --format="value(key,value)"
$G compute instances get-guest-attributes slider --zone us-central1-b $PR --query-path=slider/ --format="value(key,value)"

# 강화학습 저장 파일
$G storage ls -l gs://slider-rl-eminent-glider/

# GPU VM 이어받기 (없을 때만 띄움)
bash gcp/relaunch_rl.sh
```

SSH 는 이 작업 환경에서 막혀 있습니다 (22번 포트). 사람은 구글 클라우드 콘솔 → VM 인스턴스 → **SSH** 버튼으로 들어가서
`sudo tail -f /root/slider_rl/log.txt` (강화학습), `sudo tail -f /root/spsa.log` (SPSA), `sudo tail -f /root/queue.log` (실험) 로 볼 수 있습니다.
보기만 하고 `shutdown` / 프로세스 종료는 하지 마세요.

## 6. 긴급 정지 (전부 지우기)

```bash
G="env -u CLOUDSDK_AUTH_ACCESS_TOKEN gcloud"; PR="--project eminent-glider-465204-b2"
# 1) 결과부터 Drive 로 (GCS 에 있는 것)
$G storage cp "gs://slider-rl-eminent-glider/*" ./rl_backup/
# 2) VM 전부 삭제 (디스크도 같이 지워짐)
for v in $($G compute instances list $PR --format="value(name,zone.basename())" | tr '\t' ':'); do
  $G compute instances delete ${v%%:*} --zone ${v##*:} $PR --quiet
done
# 3) 남은 디스크/고정 IP 확인 (비어야 함)
$G compute disks list $PR; $G compute addresses list $PR
# 4) 버킷 정리 (결과를 옮긴 뒤에만)
$G storage rm -r gs://slider-rl-eminent-glider
# 5) 다 끝났으면 권한 회수
$G auth revoke --all
```

사람이 직접 할 때: 콘솔 → Compute Engine → VM 인스턴스에서 전부 선택 → 삭제. Cloud Storage → 버킷 → 삭제.

## 7. 비용

| 종류 | 시간당 (Spot, 추정) |
|---|---|
| c2d-highcpu-32 (실험, SPSA, 엔진 판) | 약 $0.55 |
| c2d-highcpu-16 | 약 $0.28 |
| n1-standard-8 + T4 (강화학습) | 약 $0.23 |
| 디스크 20~50GB, GCS 66MB | 무시할 수준 (시간당 1센트 미만) |

누적 (2026-11-03 21:10 KST 기준, 추정): **약 $4.6**
- 오전 실험 1~6차 (32코어 약 3시간 50분): $2.1
- 탐색 개선 12개 (2시간 30분): $1.4
- 규칙 변경 전 GPU + 16코어 시험 (각 35분, 중단): $0.3
- 새 규칙 확인 4개 (20분): $0.2
- 새 규칙 SPSA (진행 중), 엔진 판 (6분), GPU 강화학습 (진행 중): $0.6~

오늘 밤 계획을 다 하면 약 $8.3, 내일 GPU 12시간을 더하면 약 $11 입니다 (한도 $18.5).

## 8. 이 작업 환경이 재시작됐을 때 (세션 복구)

이 Claude 세션은 클라우드 컨테이너에서 돌고, 오래 쉬면 컨테이너가 바뀔 수 있습니다. 그러면:

- **예약 작업은 살아 있습니다** (세션에 묶여 있어서 컨테이너가 바뀌어도 깨워 줌).
- **임시 폴더의 스크립트와 gcloud 로그인은 사라질 수 있습니다.**
  - 스크립트: 레포의 `gcp/startup/`, `gcp/relaunch_rl.sh` 를 쓰면 됩니다 (`git clone -b rust-port https://github.com/Perdume/Slider.git`).
  - gcloud 로그인: `gcloud auth login --no-launch-browser` 를 다시 해야 하고, **사용자가 브라우저에서 구글 계정으로 승인**해야 합니다.
    로그인이 풀린 동안 VM 은 각자의 최대 실행 시간이 되면 스스로 지워지므로 요금이 계속 나가지는 않습니다.
    다만 Spot 회수 후 이어받기는 못 하므로, 그 사이 학습은 멈춥니다.
- 비용 장부(`~/slider_cost_ledger.txt` 또는 임시 폴더)가 사라졌으면 이 문서의 7절 숫자에서 다시 시작합니다.

## 9. 사람 손이 필요한 일

- 지금은 **없습니다** (VM 과 이 작업 환경에서 모두 처리).
- gcloud 로그인이 풀렸을 때 다시 승인 (8절).
- 폰용 튜닝 앱(APK)을 만들면 폰에 설치 (빌드는 VM 에서).
- Colab 으로 이어서 학습할 때 노트북 실행과 구글 로그인 창 승인.

## 10. 지금까지의 결론 (요약)

- 게임 설정 = lmr2 + SPSA 상수(`Tunables::game`) + 튜닝 가중치(`weights/tuned.json`) + 모든 코어 병렬 탐색. 새 규칙에서도 각각 Elo +84 / +122 / +98 로 유효.
- 기각: PVS, NMP, qs(위협 정리 탐색), 위치 점수 변경, contempt, 탐색 개선 6가지(예전 규칙), 함정 보너스 끄기.
- 새 규칙은 무승부가 0~11% (예전 70%) 라서 실력 차이가 잘 드러나고 학습 신호가 강함.
- 강화학습: 엔진 판 사전학습 후 엔진 수 맞히기 83.4%, 엔진 상대 첫 점수 0승 4무 60패 (학습 시작 전).
- 결과 기록: Drive `slider_results` (run1 ~ run9 요약 파일).

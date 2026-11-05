# Slider AI 튜닝 앱 (Android)

폰 CPU 로 **AI 의 탐색 상수와 평가 가중치를 대국 결과로 조금씩 맞추는(SPSA)** 앱입니다. 규칙은 지금 게임(새 규칙: 대각선 배치,
말 하나씩 시계 방향)이고, 시작값은 VM 에서 맞춘 게임 설정입니다. GCP 예산이 끝난 뒤에도 폰으로 튜닝을 이어가기 위한 것입니다.
실제 계산은 `rust/src/bin/spsa.rs` (arm64 실행 파일 `libspsa.so`) 가 하고, 앱은 그걸 포그라운드 서비스로 띄우는 얇은 껍데기입니다.

## 설치와 사용
1. `slider-train.apk` 를 폰에 옮겨서 설치합니다 (출처를 알 수 없는 앱 설치 허용 필요, 디버그 키로 서명됨).
2. 앱을 열고 **시작**. 화면을 꺼도 계속 돕니다 (포그라운드 서비스 + CPU 깨어 있음 유지, 최대 24시간).
   **충전기를 꽂고** 돌리세요. 뜨거워지면 스레드 수를 줄이세요.
3. 중지했다가 다시 시작하면 **저장된 진행에서 이어서** 합니다 (반복마다 저장). 처음부터 하려면 "처음부터 다시" 를 체크.
4. **결과를 다운로드 폴더에 저장** → `slider_spsa_tune_*.txt`, `slider_spsa_weights_*.json` 이 Download 폴더에 생깁니다.
   PC 에서 `matchup --a slider_spsa_weights_*.json --tune-a "$(cat slider_spsa_tune_*.txt)" --b ../weights/tuned.json --tune-b game ...`
   으로 지금 게임 설정과 비교해서, 이기면 게임에 넣습니다.

설정:
- 반복당 대국 수 (기본 32): 많을수록 한 반복이 정확하지만 느립니다.
- 수당 탐색 시간 (기본 30ms): VM 에서 쓴 값과 같게 두는 것이 좋습니다.
- 시작 탐색 상수 / 시작 가중치 (선택): 비우면 앱에 들어 있는 게임 설정에서 시작합니다. 새 값을 받았으면 여기에 붙여 넣고 "처음부터 다시".

**AI 대국 보기**: AI 끼리 한 판을 두며 매 수마다 판(말 1·2 = 먼저 두는 쪽, X·Y = 나중 쪽), 움직인 말과 방향, 형세 점수와 탐색 깊이를
로그에 보여 줍니다. "A 쪽은 튜닝 중인 값으로" 를 켜면 폰에서 튜닝한 값(A)과 게임 기본값(B)이 맞붙습니다. PC 에서는 `cargo run --release --bin watch`.

맞추는 값: 탐색 상수 11개 (margin, trap, plan_ply, lmr_depth, lmr_index, lmr2_depth, lmr2_index, nt_me, nt_opp, pos_nx, mob_oc) + 평가 가중치 11개.

## NPU / GPU 를 쓰지 않는 이유
- 시간의 대부분은 **탐색**(비트보드 이동 생성과 알파베타)이고, 정수 연산과 분기가 많아서 NPU/GPU 에 올릴 수 없습니다.
- 안드로이드 NPU 는 추론용이고 학습을 지원하는 공개 경로가 없습니다.
- 합성곱 신경망 강화학습(JAX)은 GPU 가 필요해서 Colab / GCP 에서 합니다.

## 빌드
```bash
sdkmanager "platforms;android-34" "build-tools;34.0.0" "ndk;26.3.11579264"
ANDROID_HOME=/path/to/sdk ./build_apk.sh      # 결과: build/slider-train.apk
```
Gradle 을 쓰지 않고 SDK 도구(aapt2, d8, zipalign, apksigner)만 씁니다. 실행 파일은 `lib/arm64-v8a/lib*.so` 로 APK 에 들어가고
(`extractNativeLibs=true`), 설치 때 앱의 네이티브 폴더에 풀려서 거기서 실행됩니다.

## 확인한 것과 못 한 것
- 확인: APK 서명/정렬/매니페스트, 같은 `spsa` 를 x86 에서 실행·중지·이어서 하기.
- **못 한 것: 실제 안드로이드 기기나 에뮬레이터에서의 설치와 실행.** 실행이 막히면 로그에 `Permission denied` 같은 줄이 나옵니다.

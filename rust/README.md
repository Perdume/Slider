# Slider 엔진 (Rust)

Java `org.example.Learning` 의 탐색 엔진과 자가대전/대국 도구를 Rust 로 옮긴 것입니다. 의존성이 없습니다.
콘솔 게임(`Game`)은 Java 에 그대로 있고, 이쪽은 학습 데이터 생성과 가중치 비교를 빠르게 돌리는 용도입니다.

```bash
cd rust
cargo build --release
cargo test --release

# 자가대전 데이터 (JSONL, Java SelfPlay 와 같은 형식)
target/release/selfplay --games 200 --time-ms 30 --out data/selfplay.jsonl

# 가중치 두 벌 맞붙이기 (마지막 RESULT 줄 형식도 Java Match 와 같다)
target/release/matchup --a weights.json --b default --games 100 --time-ms 50
```

옵션 이름과 기본값은 Java 도구와 같습니다 (`--games --time-ms --depth --random-plies --epsilon --max-plies --threads --seed --weights --out`).

## Java 와 같은 결과를 내는지

시작 수와 무작위 수에 `java.util.Random` 과 같은 수열을 쓰고(`javarandom.rs`) 탐색 로직도 그대로 옮겼기 때문에,
시간 제한을 없애고 깊이만 정해서 돌리면 두 구현이 똑같은 판을 둡니다. 같은 인자로 돌려서 출력 파일을 비교했습니다.

| 설정 | 판 / 국면 | 출력 파일 | Java | Rust |
|---|---|---|---|---|
| 깊이 5, 스레드 1, 시드 7 | 20 / 1,961 | 바이트 단위로 같음 | 10.2초 | 6.6초 |
| 깊이 8, 스레드 1, 시드 11 | 6 / 733 | 바이트 단위로 같음 | 29.9초 | 18.9초 |

```bash
ARGS="--games 20 --time-ms 600000 --depth 5 --threads 1 --seed 7 --epsilon 0.05"
java -cp target/classes org.example.Learning.SelfPlay $ARGS --out java.jsonl
rust/target/release/selfplay $ARGS --out rust.jsonl
cmp java.jsonl rust.jsonl
```

시간 제한이 있는 실제 대국은 탐색이 시간에 걸려서 실행마다 달라지므로 판 단위로는 재현되지 않습니다.

## 입력 표현과 국면 해시 (NNUE 방식)

`src/nnue.rs` 는 국면 `(tiles, me, opp)` 를 147개 이진 입력 중 켜진 것의 번호 목록으로 만든다
(타일 0..48, 둘 차례인 쪽 말 49..97, 반대편 말 98..146).
이동 후의 국면은 한 칸씩 고치지 않고 **결과 비트보드에서 통째로 다시 계산**한다.
국면 해시도 같은 입력으로 만든다: 켜진 입력마다 정해진 64비트 값을 XOR 한다.
Java 의 `SliderAI.positionHash` 도 같은 값을 쓰도록 바꿔서 두 구현이 계속 같은 판을 둔다.

해시를 바꾸면 트랜스포지션 테이블의 충돌 패턴이 달라져서, 한쪽만 바꾸면 깊이 8에서 결과가 갈라졌다. 양쪽을 같이 바꾸고 다시 비교했다.
속도는 이전 해시와 거의 같다 (Rust 깊이 8: 18.2초 -> 18.9초). 이 모듈의 의미는 속도보다 이후 NNUE 평가의 입력 형식을 미리 정해 두는 데 있다.
평가 함수는 아직 11개 특징의 선형식이고 신경망은 들어 있지 않다.

## 신경망 (NNUE 방식 평가)

평가 점수 = 기존 선형 평가 + 신경망 출력 × `OUT_SCALE`(200). 신경망은 147개 희소 입력 → 은닉 32개(clipped ReLU) → 출력 1개이고,
출력층이 0으로 시작해서 학습 전에는 기존 평가와 똑같다. 추론은 켜진 입력(20개 안팎)을 합쳐서 매번 새로 계산한다.
추론, 학습(역전파 + Adam), 파일 입출력을 모두 외부 라이브러리 없이 직접 구현했다 (`src/nnue.rs`, `src/bin/trainnet.rs`).

```bash
# 1) 판 정보를 포함해서 자가대전 (선형 가중치는 --weights, 이미 신경망이 있으면 --net)
target/release/selfplay --games 600 --time-ms 30 --weights weights.json --positions yes --out data/nn1.jsonl

# 2) 학습: 탐색 점수와 판 결과를 섞은 목표 (--lambda 1 이면 탐색 점수만)
target/release/trainnet --data data/nn1.jsonl --weights weights.json --out net.txt --epochs 40 --lambda 1.0

# 3) 신경망이 정말 강해졌는지 대국으로 확인 (A 가 신경망 있음)
target/release/matchup --a weights.json --net-a net.txt --b weights.json --games 200 --time-ms 50
```

Colab 에서는 `colab/slider_nnue_training.ipynb` 가 위 과정을 라운드로 반복한다 (자가대전 → 학습 → 대국 확인 → 이기면 채택).

학습 때 지킨 것:
- 역전파는 수치 미분(중앙 차분)과 대조하는 테스트로 확인했다 (`cargo test`).
- 판 단위로 80/20 분할하고, 검증 손실이 가장 낮은 시점의 파라미터를 저장한다 (외우기만 하는 학습을 걸러낸다).
- 눈금 K 는 400 으로 고정한다. 판 결과에 맞춰 보정하면 선형 평가가 승패와 거의 무관할 때 K 가 범위 끝(약 18,000)으로 가서
  sigmoid 가 평평해지고 탐색 점수 신호가 사라진다. 처음 구현에서 이 때문에 검증 손실이 계속 오르는 것(외우기)을 확인하고 고쳤다.

### 지금까지의 결과 (정직한 기록)

로컬에서 튜닝한 선형 가중치로 자가대전 305판(국면 31,673개, 수당 30ms)을 모아 한 번 학습해 봤다.

| 학습 목표 (`--lambda`) | 검증 손실 (선형 평가만 → 신경망) | 선형 평가와 100판 대국 (수당 50ms) |
|---|---|---|
| 1.0 (탐색 점수만) | 0.01191 → 0.00821 (-31%) | 7승 79무 14패, 점수 0.465 (구간 0.421~0.509) |
| 0.7 (탐색 점수 70% + 판 결과 30%) | 0.02314 → 0.02040 (-12%) | 6승 71무 23패, 점수 0.415 (구간 0.365~0.465) |

- 신경망은 탐색 점수를 잘 따라 하게 됐지만(검증 손실 감소), **이 데이터 양에서는 대국에서 선형 평가보다 강해지지 않았다.** 판 결과를 섞으면 오히려 약해졌다.
- 평가 점수와 판 결과의 상관이 거의 0(0.016)이라서 판 결과는 노이즈가 많은 목표다. 그래서 기본값은 `--lambda 1.0`.
- 대국에서 지는 이유는 알아내지 못했다. 데이터가 더 많으면 달라질 수도, 아닐 수도 있다.
  노트북의 채택 기준(대국 점수 0.55 이상)이 이런 신경망을 걸러 낸다.

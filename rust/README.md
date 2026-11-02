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

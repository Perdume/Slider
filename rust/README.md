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
| 깊이 5, 스레드 1, 시드 7 | 20 / 1,961 | 바이트 단위로 같음 | 11.5초 | 6.3초 |
| 깊이 8, 스레드 1, 시드 11 | 6 / 733 | 바이트 단위로 같음 | 27.5초 | 18.2초 |

```bash
ARGS="--games 20 --time-ms 600000 --depth 5 --threads 1 --seed 7 --epsilon 0.05"
java -cp target/classes org.example.Learning.SelfPlay $ARGS --out java.jsonl
rust/target/release/selfplay $ARGS --out rust.jsonl
cmp java.jsonl rust.jsonl
```

시간 제한이 있는 실제 대국은 탐색이 시간에 걸려서 실행마다 달라지므로 판 단위로는 재현되지 않습니다.

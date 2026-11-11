# Slider

7x7 보드에서 타일을 타고 미끄러져 중앙(골)에 먼저 멈추는 쪽이 이기는 콘솔 게임입니다.
플레이어와 AI가 **말 2개씩** 가지고 번갈아 둡니다.

## 규칙

- 보드: 7x7, 타일(■) 12개가 네 귀퉁이에 3개씩 있습니다. 골(G)은 (3,3)입니다.
- 말: 플레이어는 위쪽 두 귀퉁이 (0,0), (0,6)에서 `1`, `2`로, AI는 아래쪽 두 귀퉁이 (6,0), (6,6)에서 `X`, `Y`로 시작합니다. 180도 돌리면 서로 겹치는 공정한 배치입니다.
- **한 턴에 말 하나만** 골라서 움직입니다. 먼저 `1` 또는 `2`로 말을 고른 뒤 `w`(위) `a`(왼쪽) `s`(아래) `d`(오른쪽)를 입력합니다.
  - 옆 칸이 타일이면 그 타일로 **걸어갑니다**. 턴은 계속됩니다.
  - 옆 칸이 빈 칸이면 서 있는 타일째 그 방향으로 **미끄러집니다**. 다른 타일이나 벽 바로 앞에서 멈추고 턴이 끝납니다.
  - 다른 말이 있는 칸(상대 말, 내 다른 말)으로는 갈 수 없습니다.
  - 걷다가 막혔거나 마음이 바뀌면 `r`을 입력해 그 턴을 말 고르기부터 다시 합니다.
- **어느 말이든** 골에 정확히 멈추면 이깁니다. 미끄러짐은 타일/벽 앞에서만 멈추므로, 골 건너편에 **받침 타일**이 있어야 합니다.
- 둘 수 있는 수가 없으면(두 말 모두 갇힘) 차례가 넘어갑니다.
- 같은 국면(누구 차례인지 포함)이 3번 나오면 무승부입니다.
- 플레이어가 먼저 둡니다.

## 실행

기본 엔진은 Rust 입니다 (같은 규칙, Java 보다 빠름, 병렬 탐색 + LMR + 튜닝 가중치가 기본).

```bash
cd rust && cargo run --release --bin play            # 옵션: --time-ms 1500 --threads N --search lmr2|lmr|none
```

Java 판(예전 구현)도 그대로 실행할 수 있습니다.

```bash
mvn compile
java -cp target/classes org.example.Main
```

테스트:

```bash
mvn test
```

## AI

AI는 한 수에 약 1.5초 동안 알파베타 탐색을 합니다.
기본으로 **병렬 탐색**(컴퓨터의 코어를 모두 써서 한 수를 함께 탐색, Lazy SMP)과 **LMR**(뒤쪽 수를 얕게 먼저 보는 가지치기)을 씁니다.
Rust 엔진으로 시간 제한 대국을 SPRT 로 검증한 개선입니다 (병렬 탐색 4스레드 Elo +100, 더 센 LMR 약 Elo +57).
끄거나 바꾸려면: `java -Dslider.threads=1 -Dslider.search=none -cp target/classes org.example.Main` (`slider.search` 는 `none`, `lmr`, `lmr2`). 구조, 평가 함수, 성능 측정 결과, 다음 계획은
[docs/AI.md](docs/AI.md)에 정리되어 있습니다.

### 학습 (Colab)

평가 함수의 가중치를 자가대전 데이터로 튜닝하는 노트북이 `colab/slider_2v2_training.ipynb`에 있습니다.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Perdume/Slider/blob/master/colab/slider_2v2_training.ipynb)

노트북이 하는 일: 자가대전 데이터 생성(`SelfPlay`) → 가중치 학습 → 새 가중치 vs 현재 가중치 대국(`Match`) → `weights.json` 저장.
결과 가중치는 이렇게 적용합니다.

```bash
java -Dslider.weights=weights.json -cp target/classes org.example.Main
```

Java 쪽 도구는 노트북 없이 직접 실행할 수도 있습니다.

```bash
# 자가대전 데이터 생성 (JSONL)
java -cp target/classes org.example.Learning.SelfPlay --games 200 --time-ms 30 --out data/selfplay.jsonl

# 가중치 두 벌 맞붙이기 (default 는 기본 가중치)
java -cp target/classes org.example.Learning.Match --a weights.json --b default --games 100 --time-ms 50
```

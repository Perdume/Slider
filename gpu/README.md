# Slider 규칙 (JAX) — GPU 에서 많은 판을 동시에

`slider_jax.py` 는 게임 규칙만 JAX 배열 연산으로 옮긴 것이다 (탐색/신경망은 아직 없다). `vmap` 으로 수만 판을 한 번에 둘 수 있다.
수 선택은 정책 함수로 끼워 넣는다 (`play_game(policy, key)`). 지금은 무작위와 "ply*a+b 번째 수" 두 가지만 있다.

## Rust 와 같은 규칙인지 확인
```bash
cd rust && cargo build --release --example rules_dump && cd ../gpu
pip install jax
python check_rules.py ../rust/target/release/examples/rules_dump
```
확인한 것 (모두 일치):
- 시작 국면에서 깊이 6까지 노드 수 (8, 58, 362, 2254, 16560, 121472)
- 무작위 대국에서 뽑은 국면 43,955개의 가능한 수 목록 (집합이 같음)
- 정해진 규칙으로 둔 400판의 승패와 수 (골 도달, 통과, 3회 반복 무승부, 150수 무승부 포함)

## 속도
`python bench.py` (Colab GPU 용 노트북: `colab/slider_gpu_bench.ipynb`).

| 환경 | 무작위 대국 처리 속도 |
|---|---|
| Rust, CPU 한 코어 | 약 4,100,000 수/초 |
| JAX, 이 개발 환경 CPU 2코어 (배치 1024) | 약 47,000 수/초 |
| JAX, GPU | **아직 측정 못 함** (개발 환경에 GPU 없음, Colab 에서 측정 필요) |

GPU 가 이 속도 비교에서 Rust CPU 코어 몇 개 값을 하는지는 측정해 봐야 안다.

## 아직 없는 것
- 탐색 (알파베타는 GPU 에 안 맞아서, 배치 고정 깊이 탐색이나 MCTS 로 새로 만들어야 한다)
- 신경망 평가, 학습 루프

//! Slider 2v2 엔진의 Rust 포팅. Java 구현(`src/main/java/org/example/Learning`)과 같은 규칙과 같은 탐색을 한다.
//!
//! - `bitboard`: 7x7 비트보드와 이동 생성
//! - `weights`: 평가 가중치 (JSON 읽기/쓰기)
//! - `ai`: 반복 심화 negamax 탐색과 평가 함수
//! - `arena`: AI 끼리 2v2 한 판을 두는 공용 코드
//! - `javarandom`: java.util.Random 과 같은 난수 (시작 수를 Java 와 똑같이 재현하기 위해)

pub mod ai;
pub mod arena;
pub mod bitboard;
pub mod cli;
pub mod javarandom;
pub mod weights;

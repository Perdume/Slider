//! Java 구현이 뽑은 기준 데이터(tests/golden.txt)와 Rust 구현의 결과를 대조한다.
//! 같은 국면에서 이동 생성 순서, 평가 특징, 정해진 깊이 탐색의 (수, 점수, 끝낸 깊이)가 모두 같아야 한다.
//! 기준 데이터는 tools/GoldenGen.java 로 다시 만들 수 있다.

use std::collections::HashMap;

use slider::ai::SliderAi;
use slider::bitboard::{self as bb, MAX_MOVES};
use slider::weights;

struct Case {
    line: usize,
    tiles: u64,
    me: u64,
    opp: u64,
    depth: i32,
    history: HashMap<u64, u32>,
    moves: Vec<i32>,
    best_move: i32,
    score: i32,
    reached_depth: i32,
    features: Vec<i32>,
}

fn parse(text: &str) -> Vec<Case> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let t: Vec<i64> = l.split_whitespace().map(|x| x.parse().expect("숫자")).collect();
            let mut p = 0;
            let mut take = || {
                p += 1;
                t[p - 1]
            };
            let tiles = take() as u64;
            let me = take() as u64;
            let opp = take() as u64;
            let depth = take() as i32;
            let n_hist = take() as usize;
            let mut history = HashMap::new();
            for _ in 0..n_hist {
                let h = take() as u64;
                let c = take() as u32;
                history.insert(h, c);
            }
            let n_moves = take() as usize;
            let moves = (0..n_moves).map(|_| take() as i32).collect();
            let best_move = take() as i32;
            let score = take() as i32;
            let reached_depth = take() as i32;
            let features = (0..weights::COUNT).map(|_| take() as i32).collect();
            Case { line: i + 1, tiles, me, opp, depth, history, moves, best_move, score, reached_depth, features }
        })
        .collect()
}

fn cases() -> Vec<Case> {
    parse(include_str!("golden.txt"))
}

#[test]
fn fixture_is_not_empty() {
    assert!(cases().len() >= 80);
}

#[test]
fn move_generation_matches_java_including_order() {
    for c in cases() {
        let mut out = vec![0i32; MAX_MOVES];
        let (n, _) = bb::generate(c.tiles, c.me, c.opp, &mut out);
        assert_eq!(&out[..n], &c.moves[..], "line {}", c.line);
    }
}

#[test]
fn features_match_java() {
    let mut checked = 0;
    for c in cases() {
        let wins_now = c.moves.iter().any(|&m| bb::dest(m) == bb::GOAL);
        if c.moves.is_empty() || wins_now {
            continue;
        }
        let mut ai = SliderAi::new(10, 2);
        let f = ai.features(c.tiles, c.me, c.opp);
        assert_eq!(&f[..], &c.features[..], "line {}", c.line);
        checked += 1;
    }
    assert!(checked > 50, "특징을 비교한 국면이 너무 적습니다: {}", checked);
}

#[test]
fn fixed_depth_search_matches_java() {
    for c in cases() {
        let mut ai = SliderAi::new(600_000, c.depth);
        ai.set_game_history(c.history.clone());
        let r = ai.search(c.tiles, c.me, c.opp);
        assert_eq!(
            (r.mv, r.score, r.depth),
            (c.best_move, c.score, c.reached_depth),
            "line {} (depth {}): Rust (수, 점수, 깊이) 가 Java 와 다릅니다",
            c.line,
            c.depth
        );
    }
}

#[test]
fn fixture_covers_pass_history_and_immediate_win() {
    let all = cases();
    assert!(all.iter().any(|c| c.moves.is_empty() && c.best_move == -1), "차례 넘김 국면이 없습니다");
    assert!(all.iter().any(|c| !c.history.is_empty()), "반복 이력이 있는 국면이 없습니다");
    assert!(all.iter().any(|c| c.moves.iter().any(|&m| bb::dest(m) == bb::GOAL)), "즉시 승리 국면이 없습니다");
}

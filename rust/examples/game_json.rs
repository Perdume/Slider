//! AI 끼리 둔 판을 JSON 으로 기록한다 (웹 뷰어용). 새 규칙(순환), 게임 설정.
//!   cargo run --release --example game_json -- <판 수> <수당 ms> <스레드> <시작 seed> > games.json
use slider::ai::{generate_turn, turn_hash, History, SearchOptions, SliderAi, Tunables, WIN_BOUND};
use slider::arena::{CYCLE_SIDE0_START, CYCLE_SIDE1_START, START_BOARD};
use slider::bitboard::{self as bb, GOAL, SIZE};
use slider::weights::EvalWeights;

const TUNED: &str = include_str!("../../weights/tuned.json");
const DELTA: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // w s a d

fn game(time_ms: u64, threads: usize, seed0: u64, random_plies: usize) -> String {
    let w = EvalWeights::from_json(TUNED).unwrap();
    let opts = SearchOptions::parse("lmr2,hm").unwrap();
    let start = |l: &[(usize, usize); 2]| [l[0].0 * SIZE + l[0].1, l[1].0 * SIZE + l[1].1];
    let mut cells = [start(&CYCLE_SIDE0_START), start(&CYCLE_SIDE1_START)];
    let mut next = [0usize, 0usize];
    let mut tiles = bb::from_array(&START_BOARD);
    let mask = |c: &[usize; 2]| (1u64 << c[0]) | (1u64 << c[1]);
    let mut history = History::default();
    let mut buf = vec![0i32; bb::MAX_MOVES];
    let mut seed = seed0.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut plies = Vec::new();
    let mut result = "draw".to_string();
    let mut reason = "400수 제한".to_string();
    for ply in 0..400 {
        let side = ply & 1;
        let (me, opp) = (mask(&cells[side]), mask(&cells[1 - side]));
        let (mc, oc) = (1u64 << cells[side][next[side]], 1u64 << cells[1 - side][next[1 - side]]);
        let h = turn_hash(tiles, me, opp, mc, oc);
        *history.entry(h).or_insert(0) += 1;
        if history[&h] >= 3 {
            reason = "같은 국면 3번 반복".into();
            break;
        }
        let (n, _) = generate_turn(tiles, me, opp, mc, &mut buf);
        if n == 0 {
            plies.push(format!("{{\"side\":{},\"piece\":{},\"pass\":true}}", side, next[side]));
            next[side] ^= 1;
            continue;
        }
        let (mv, score, depth, nodes, random) = if ply < random_plies {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (buf[(seed % n as u64) as usize], 0, 0, 0, true)
        } else {
            let mut ai = SliderAi::with_weights(time_ms, 40, &w);
            ai.set_options(opts);
            ai.set_tunables(Tunables::game());
            ai.set_threads(threads);
            ai.set_game_history(history.clone());
            let r = ai.search_turn(tiles, me, opp, mc, oc);
            (r.mv, r.score, r.depth, r.nodes, false)
        };
        let syms = bb::to_symbols(tiles, me, opp, mv);
        let mut path = vec![bb::src(mv)];
        let (mut r, mut c) = ((bb::src(mv) / SIZE) as i32, (bb::src(mv) % SIZE) as i32);
        for s in &syms[..syms.len() - 1] {
            let d = ["w", "s", "a", "d"].iter().position(|x| x == s).unwrap();
            r += DELTA[d].0;
            c += DELTA[d].1;
            path.push((r * SIZE as i32 + c) as usize);
        }
        let slide_dir = ["w", "s", "a", "d"].iter().position(|x| *x == syms[syms.len() - 1]).unwrap();
        tiles = bb::apply(tiles, mv);
        cells[side][next[side]] = bb::dest(mv);
        let p: Vec<String> = path.iter().map(|x| x.to_string()).collect();
        plies.push(format!(
            "{{\"side\":{},\"piece\":{},\"path\":[{}],\"from\":{},\"dir\":{},\"dest\":{},\"tiles\":\"{}\",\"score\":{},\"depth\":{},\"nodes\":{},\"random\":{}}}",
            side, next[side], p.join(","), bb::from(mv), slide_dir, bb::dest(mv), tiles, score.clamp(-WIN_BOUND - 1, WIN_BOUND + 1), depth, nodes, random
        ));
        if bb::dest(mv) == GOAL {
            result = if side == 0 { "A" } else { "B" }.into();
            reason = "골 도착".into();
            break;
        }
        next[side] ^= 1;
    }
    let _ = WIN_BOUND;
    format!("{{\"seed\":{},\"timeMs\":{},\"result\":\"{}\",\"reason\":\"{}\",\"plies\":[{}]}}", seed0, time_ms, result, reason, plies.join(","))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let games: u64 = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(3);
    let time_ms: u64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let threads: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(2);
    let seed: u64 = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut out = Vec::new();
    for g in 0..games {
        eprintln!("판 {} 시작", g + 1);
        out.push(game(time_ms, threads, seed + g, 2));
    }
    println!("{{\"goal\":{},\"start\":\"{}\",\"startCells\":[[{},{}],[{},{}]],\"winBound\":{},\"games\":[{}]}}",
        GOAL, bb::from_array(&START_BOARD),
        CYCLE_SIDE0_START[0].0 * SIZE + CYCLE_SIDE0_START[0].1, CYCLE_SIDE0_START[1].0 * SIZE + CYCLE_SIDE0_START[1].1,
        CYCLE_SIDE1_START[0].0 * SIZE + CYCLE_SIDE1_START[0].1, CYCLE_SIDE1_START[1].0 * SIZE + CYCLE_SIDE1_START[1].1,
        WIN_BOUND, out.join(","));
}

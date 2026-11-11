//! AI 끼리 두는 한 판을 한 수씩 보여 준다 (새 규칙: 대각선 배치, 말 하나씩 시계 방향).
//!
//!   cargo run --release --bin watch
//! 옵션: --time-ms 500 (수당 탐색 시간) --delay-ms 800 (수 사이 쉬는 시간) --threads N (한 수에 쓰는 스레드, 기본 코어 수)
//!       --tune-a / --tune-b (기본 game) --weights-a / --weights-b (기본 tuned 또는 JSON 경로) --search lmr2,hm
//!       --random-plies 0 (처음 몇 수를 무작위로, 판을 다양하게) --seed 1 --max-plies 400 --stop-file 경로
//! A = 먼저 두는 쪽 (말 1, 2), B = 나중 쪽 (말 X, Y). 차례: 1 -> X -> 2 -> Y -> ...
use std::io::Write;
use std::time::{Duration, Instant};

use slider::ai::{generate_turn, turn_hash, History, SearchOptions, SliderAi, Tunables, WIN_BOUND};
use slider::arena::{CYCLE_SIDE0_START, CYCLE_SIDE1_START, START_BOARD};
use slider::bitboard::{self as bb, GOAL, SIZE};
use slider::cli;
use slider::weights::EvalWeights;

const TUNED: &str = include_str!("../../../weights/tuned.json");
const MARKS: [[char; 2]; 2] = [['1', '2'], ['X', 'Y']];
const DIR_NAME: [(&str, &str); 4] = [("w", "위"), ("s", "아래"), ("a", "왼쪽"), ("d", "오른쪽")];

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

fn weights_arg(v: Option<&String>) -> Result<EvalWeights, String> {
    match v.map(String::as_str) {
        None | Some("tuned") => EvalWeights::from_json(TUNED),
        Some(p) => EvalWeights::from_argument(Some(p)),
    }
}

fn tune_arg(v: Option<&String>) -> Result<Tunables, String> {
    match v.map(String::as_str) {
        None | Some("game") => Ok(Tunables::game()),
        Some(t) => Tunables::parse(t),
    }
}

fn board(tiles: u64, cells: &[[usize; 2]; 2], next: [usize; 2], to_move: usize) -> String {
    let mut out = String::new();
    for r in 0..SIZE {
        out.push_str("  ");
        for c in 0..SIZE {
            let cell = r * SIZE + c;
            let mut mark = if bb::is_tile(tiles, cell) { '■' } else if cell == GOAL { 'G' } else { '□' };
            for s in 0..2 {
                for i in 0..2 {
                    if cells[s][i] == cell {
                        mark = MARKS[s][i];
                    }
                }
            }
            out.push(mark);
            out.push(' ');
        }
        out.push('\n');
    }
    let other = 1 - to_move;
    out.push_str(&format!("  다음: {} -> {}", MARKS[to_move][next[to_move]], MARKS[other][next[other]]));
    out
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let time_ms: u64 = cli::get(&opt, "time-ms", 500)?;
    let delay_ms: u64 = cli::get(&opt, "delay-ms", 800)?;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads())?.max(1);
    let max_plies: usize = cli::get(&opt, "max-plies", 400)?;
    let random_plies: usize = cli::get(&opt, "random-plies", 0)?;
    let mut seed: u64 = cli::get::<u64>(&opt, "seed", 1)?.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let opts = SearchOptions::parse(opt.get("search").map(String::as_str).unwrap_or("lmr2,hm"))?;
    let stop = cli::Stop::from_options(&opt)?;
    let w = [weights_arg(opt.get("weights-a"))?, weights_arg(opt.get("weights-b"))?];
    let tune = [tune_arg(opt.get("tune-a"))?, tune_arg(opt.get("tune-b"))?];
    let same = w[0] == w[1] && tune[0] == tune[1];
    println!("AI 대국 보기 (새 규칙): 수당 {}ms, 스레드 {}, 탐색 {}{}", time_ms, threads, opts, if same { ", 양쪽 같은 설정" } else { ", A/B 설정 다름" });

    let start = |l: &[(usize, usize); 2]| [l[0].0 * SIZE + l[0].1, l[1].0 * SIZE + l[1].1];
    let mut cells = [start(&CYCLE_SIDE0_START), start(&CYCLE_SIDE1_START)];
    let mut next = [0usize, 0usize];
    let mut tiles = bb::from_array(&START_BOARD);
    let mask = |c: &[usize; 2]| (1u64 << c[0]) | (1u64 << c[1]);
    let mut history = History::default();
    let mut buf = vec![0i32; bb::MAX_MOVES];
    println!("{}\n", board(tiles, &cells, next, 0));

    for ply in 0..max_plies {
        if stop.should_stop() {
            println!("중지했습니다 ({}수까지)", ply);
            return Ok(());
        }
        let side = ply & 1;
        let (me, opp) = (mask(&cells[side]), mask(&cells[1 - side]));
        let (mc, oc) = (1u64 << cells[side][next[side]], 1u64 << cells[1 - side][next[1 - side]]);
        *history.entry(turn_hash(tiles, me, opp, mc, oc)).or_insert(0) += 1;
        if history[&turn_hash(tiles, me, opp, mc, oc)] >= 3 {
            println!("같은 국면이 3번 나와서 무승부 ({}수)", ply);
            return Ok(());
        }
        let who = format!("{} {}", if side == 0 { "A" } else { "B" }, MARKS[side][next[side]]);
        let (n, _) = generate_turn(tiles, me, opp, mc, &mut buf);
        if n == 0 {
            println!("{}수: {} 는 움직일 수 없어 넘김\n", ply + 1, who);
            next[side] ^= 1;
            continue;
        }
        let t0 = Instant::now();
        let (mv, score, depth) = if ply < random_plies {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (buf[(seed % n as u64) as usize], 0, 0)
        } else {
            let mut ai = SliderAi::with_weights(time_ms, 40, &w[side]);
            ai.set_options(opts);
            ai.set_tunables(tune[side]);
            ai.set_threads(threads);
            ai.set_game_history(history.clone());
            let r = ai.search_turn(tiles, me, opp, mc, oc);
            (r.mv, r.score, r.depth)
        };
        let path: Vec<&str> = bb::to_symbols(tiles, me, opp, mv)
            .into_iter()
            .map(|s| DIR_NAME.iter().find(|(k, _)| *k == s).map(|(_, v)| *v).unwrap_or(s))
            .collect();
        tiles = bb::apply(tiles, mv);
        cells[side][next[side]] = bb::dest(mv);
        let judge = if ply < random_plies {
            "무작위 수".to_string()
        } else if score >= WIN_BOUND {
            "이김 확정".to_string()
        } else if score <= -WIN_BOUND {
            "짐 확정".to_string()
        } else {
            format!("형세 {:+} (둔 쪽 기준), 깊이 {}", score, depth)
        };
        println!("{}수: {} 이동 [{}] | {} | {}ms", ply + 1, who, path.join(" "), judge, t0.elapsed().as_millis());
        if bb::dest(mv) == GOAL {
            println!("{}\n\n{} 이 골에 도착해서 {} 쪽 승리 ({}수)", board(tiles, &cells, next, 1 - side), who, if side == 0 { "A" } else { "B" }, ply + 1);
            return Ok(());
        }
        next[side] ^= 1;
        println!("{}\n", board(tiles, &cells, next, 1 - side));
        std::io::stdout().flush().ok();
        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
    }
    println!("{}수까지 끝나지 않아 무승부", max_plies);
    Ok(())
}

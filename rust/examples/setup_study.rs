//! 배치 단계 실험: 판 길이, 선후공 균형, 시작 형세 점수와 결과의 관계.
//!   cargo run --release --example setup_study -- baseline <판 수> <수당 ms> <스레드> <seed>
//!   cargo run --release --example setup_study -- random   <판 수> <수당 ms> <스레드> <seed>
//! 한 줄(TSV): 판 번호, 시작 점수(A 기준, 300ms), 승자(-1/0/1), 수, 수 제한 여부, 배치(타일 비트, 말 칸 4개)
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use slider::ai::{generate_turn, next_turn_pieces};
use slider::arena::Rules;
use slider::bitboard::{self as bb, MAX_MOVES};
use slider::setup::{eval_start, play_from, random_fill, Rng, Setup};

const MAX_PLIES: usize = 1500;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mode = a[0].clone();
    let games: usize = a[1].parse().unwrap();
    let ms: u64 = a[2].parse().unwrap();
    let threads: usize = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let next = AtomicUsize::new(0);
    let out = Mutex::new(());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::SeqCst);
                if g >= games {
                    break;
                }
                let mut rng = Rng::new(seed * 1_000_003 + g as u64);
                let (start, desc) = if mode == "baseline" {
                    // 지금 판에서 무작위 2수 (A, B 한 수씩) 뒤부터
                    let (mut tiles, mut p, mut c) = Rules::Cycle.start();
                    let mut buf = vec![0i32; MAX_MOVES];
                    for side in 0..2 {
                        let (n, _) = generate_turn(tiles, p[side], p[1 - side], c[side], &mut buf);
                        let mv = buf[rng.below(n)];
                        tiles = bb::apply(tiles, mv);
                        let after = bb::apply_pieces(p[side], mv);
                        c[side] = next_turn_pieces(p[side], c[side], after);
                        p[side] = after;
                    }
                    ((tiles, p, c), String::from("standard"))
                } else {
                    let mut s = Setup::default();
                    random_fill(&mut s, &mut rng);
                    let d = format!("{}\t{},{},{},{}", s.tiles, s.pieces[0][0], s.pieces[0][1], s.pieces[1][0], s.pieces[1][1]);
                    (s.start(), d)
                };
                let score = eval_start(&start, 300);
                let r = play_from(&start, ms, MAX_PLIES);
                let _l = out.lock().unwrap();
                println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", g, score, r.winner, r.plies, r.capped as u8, r.passes, desc, r.judged as u8);
                if std::env::var("SHOW").is_ok() { let mut s = Setup::default(); let _ = &mut s; }
            });
        }
    });
}

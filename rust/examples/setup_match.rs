//! 배치 AI 실험. 배치를 고르는 AI(S) 와 무작위로 놓는 쪽(R) 을 맞붙이거나(sr), S 끼리 둔다(ss).
//! 대국 단계는 두 쪽 모두 같은 게임 설정 엔진이라, 승률 차이는 배치에서만 나온다.
//!   cargo run --release --example setup_match -- <sr|ss> <판 수> <대국 ms> <평가 ms> <후보 수> <채우기 수> <스레드> <seed>
//! S 의 배치: 놓을 수 있는 칸 중 후보 몇 개를 뽑아, 남은 배치를 무작위로 채운 국면을 엔진으로 평가한 평균이 가장 좋은 칸.
//! 스왑: B 의 마지막 차례(말 Y)에서, B 쪽 기대값이 0.5 보다 낮으면 편을 바꾼다 (기다릴수록 정보가 많아 마지막이 가장 유리).
//! 한 줄(TSV): 판, S 가 처음 맡은 쪽(ss 면 -), 스왑 여부, 시작 점수(A 기준), 승자 쪽, 승자 사람(0=처음 A 를 맡은 사람), 수, 수 제한, 넘김
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use slider::setup::{eval_start, play_from, random_fill, Rng, Setup, STEPS};

fn value(s: &Setup, side: usize, eval_ms: u64, k: usize, rng: &mut Rng) -> f64 {
    let mut sum = 0.0;
    for _ in 0..k {
        let mut t = s.clone();
        random_fill(&mut t, rng);
        let sc = eval_start(&t.start(), eval_ms) as f64;
        let sc = sc.clamp(-1500.0, 1500.0);
        let pa = 1.0 / (1.0 + (-sc / 150.0).exp());
        sum += if side == 0 { pa } else { 1.0 - pa };
    }
    sum / k as f64
}

/// (고른 칸, 그 칸의 기대값) 기대값은 지금 놓는 쪽 기준
fn choose(s: &Setup, eval_ms: u64, cand: usize, k: usize, rng: &mut Rng) -> (usize, f64) {
    let mut legal = s.legal();
    for i in (1..legal.len()).rev() {
        legal.swap(i, rng.below(i + 1));
    }
    legal.truncate(cand.max(1));
    let side = s.side();
    let mut best = (legal[0], -1.0);
    for &c in &legal {
        let mut t = s.clone();
        t.place(c);
        let v = value(&t, side, eval_ms, k, rng);
        if v > best.1 {
            best = (c, v);
        }
    }
    best
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mode = a[0].clone();
    let games: usize = a[1].parse().unwrap();
    let ms: u64 = a[2].parse().unwrap();
    let eval_ms: u64 = a[3].parse().unwrap();
    let cand: usize = a[4].parse().unwrap();
    let k: usize = a[5].parse().unwrap();
    let threads: usize = a[6].parse().unwrap();
    let seed: u64 = a[7].parse().unwrap();
    let choose_end = std::env::var("CHOOSE").map(|v| v == "1").unwrap_or(false);
    let max_plies: usize = std::env::var("MAX_PLIES").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
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
                // smart[사람] = 그 사람이 배치 AI 인지
                let s_person = g % 2;
                let smart = if mode == "ss" { [true, true] } else { [s_person == 0, s_person == 1] };
                let mut s = Setup::default();
                while !s.done() {
                    let person = s.owner[s.side()];
                    if smart[person] {
                        let (c, v) = choose(&s, eval_ms, cand, k, &mut rng);
                        if !choose_end && s.can_swap() && s.step == STEPS - 1 && v < 0.5 {
                            s.swap();
                            continue; // 상대가 이어서 B 로 놓는다
                        }
                        s.place(c);
                    } else {
                        let l = s.legal();
                        s.place(l[rng.below(l.len())]);
                    }
                }
                // CHOOSE=1: 자르고 고르기. 배치가 끝난 뒤, 마지막에 놓지 않은 사람(지금 A 를 맡은 사람)이 편을 고른다
                if choose_end {
                    let chooser = s.owner[0];
                    let take_b = if smart[chooser] {
                        let sc = eval_start(&s.start(), 300) as f64;
                        sc < 0.0
                    } else {
                        rng.below(2) == 1
                    };
                    if take_b {
                        s.owner.swap(0, 1);
                        s.swap_used = true;
                    }
                }
                let start = s.start();
                let score = eval_start(&start, 300);
                let r = play_from(&start, ms, max_plies);
                let wp = if r.winner < 0 { -1 } else { s.owner[r.winner as usize] as i32 };
                let sp = if mode == "ss" { "-".to_string() } else { s_person.to_string() };
                let _l = out.lock().unwrap();
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{},{},{},{}\t{}",
                    g, sp, s.swap_used as u8, score, r.winner, wp, r.plies, r.capped as u8, r.passes, s.tiles,
                    s.pieces[0][0], s.pieces[0][1], s.pieces[1][0], s.pieces[1][1], r.judged as u8
                );
            });
        }
    });
}

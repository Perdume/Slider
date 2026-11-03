//! SPSA 튜닝 (새 규칙): 탐색 상수 + 평가 가중치를 대국 결과로 조금씩 맞춘다. gcp/spsa_cycle.py 와 같은 방식을
//! 한 프로그램으로 만든 것 (폰 앱 / PC 에서 VM 없이 이어서 돌리기용).
//!
//!   spsa --out 폴더 [--iters 100000] [--games 32] [--time-ms 30] [--threads 코어 수] [--lr 1.5]
//!        [--tune game|이름=값,...] [--weights tuned|경로] [--seed 7] [--stop-file 경로] [--parent-pid N]
//!
//! - 반복마다 폴더/spsa_state.txt 에 저장하므로, 멈췄다가 같은 --out 으로 다시 실행하면 이어서 한다.
//! - 지금 값: 폴더/spsa_tune.txt (matchup/play 의 --tune 에 그대로), 폴더/spsa_weights.json (--weights 에 그대로).
//! - 한 반복 = 상수를 무작위 부호로 +c / -c 흔든 두 엔진이 --games 판(쪽 바꿔 둠)을 두고, 이긴 쪽으로 조금 옮긴다.
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use slider::ai::{SearchOptions, Tunables};
use slider::arena::{self, Config, Player, Rules};
use slider::cli;
use slider::weights::{EvalWeights, COUNT, NAMES};

const TUNED: &str = include_str!("../../../weights/tuned.json");

#[allow(dead_code)]
struct Param {
    name: String,
    c: f64,
    lo: f64,
    hi: f64,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

fn tune_field(t: &Tunables, name: &str) -> i32 {
    match name {
        "margin" => t.margin,
        "trap" => t.trap,
        "plan_ply" => t.plan_ply,
        "lmr_depth" => t.lmr_depth,
        "lmr_index" => t.lmr_index,
        "lmr2_depth" => t.lmr2_depth,
        "lmr2_index" => t.lmr2_index,
        "nt_me" => t.nt_me,
        "nt_opp" => t.nt_opp,
        _ => 0,
    }
}

const T_DEFS: [(&str, f64, f64, f64); 9] = [
    ("margin", 8.0, 10.0, 160.0),
    ("trap", 12.0, 0.0, 200.0),
    ("plan_ply", 1.0, 3.0, 16.0),
    ("lmr_depth", 1.0, 1.0, 6.0),
    ("lmr_index", 1.0, 1.0, 8.0),
    ("lmr2_depth", 1.0, 2.0, 8.0),
    ("lmr2_index", 1.0, 1.0, 12.0),
    ("nt_me", 40.0, -600.0, 600.0),
    ("nt_opp", 40.0, -600.0, 600.0),
];

fn tune_str(theta: &[f64]) -> String {
    T_DEFS.iter().zip(theta).map(|((n, ..), v)| format!("{}={}", n, v.round() as i32)).collect::<Vec<_>>().join(",")
}

fn weights_of(theta: &[f64]) -> EvalWeights {
    let mut v = [0i32; COUNT];
    for i in 0..COUNT {
        v[i] = theta[T_DEFS.len() + i].round() as i32;
    }
    EvalWeights::of(v)
}

/// 아주 단순한 난수 (반복 번호로 시드를 정해서 이어서 돌려도 같은 흔들기)
fn signs(seed: u64, n: usize) -> Vec<f64> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let out = PathBuf::from(opt.get("out").cloned().unwrap_or_else(|| "spsa_out".into()));
    fs::create_dir_all(&out).map_err(|e| format!("{}: {}", out.display(), e))?;
    let iters: usize = cli::get(&opt, "iters", 100_000)?;
    let games: usize = cli::get::<usize>(&opt, "games", 32)?.max(2) / 2 * 2;
    let time_ms: u64 = cli::get(&opt, "time-ms", 30)?;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads())?.max(1);
    let lr: f64 = cli::get(&opt, "lr", 1.5)?;
    let seed: u64 = cli::get(&opt, "seed", 7)?;
    let stop = cli::Stop::from_options(&opt)?;
    let tune0 = match opt.get("tune").map(String::as_str) {
        None | Some("game") => Tunables::game(),
        Some(t) => Tunables::parse(t)?,
    };
    let w0 = match opt.get("weights").map(String::as_str) {
        None | Some("tuned") => EvalWeights::from_json(TUNED)?,
        Some(p) => EvalWeights::from_argument(Some(p))?,
    };

    let mut params: Vec<Param> = T_DEFS.iter().map(|(n, c, lo, hi)| Param { name: n.to_string(), c: *c, lo: *lo, hi: *hi }).collect();
    let mut start: Vec<f64> = T_DEFS.iter().map(|(n, ..)| tune_field(&tune0, n) as f64).collect();
    for i in 0..COUNT {
        let w = w0.values[i] as f64;
        let span = (3.0 * w.abs()).max(200.0);
        params.push(Param { name: NAMES[i].to_string(), c: (0.15 * w.abs()).max(4.0), lo: w - span, hi: w + span });
        start.push(w);
    }

    // 이어서 하기
    let state_path = out.join("spsa_state.txt");
    let (mut iter, mut theta) = match fs::read_to_string(&state_path) {
        Ok(text) => {
            let mut lines = text.lines();
            let it: usize = lines.next().and_then(|l| l.trim().parse().ok()).unwrap_or(0);
            let th: Vec<f64> = lines.next().unwrap_or("").split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if th.len() == params.len() {
                println!("이어서: 반복 {} 부터", it);
                (it, th)
            } else {
                (0, start.clone())
            }
        }
        Err(_) => (0, start.clone()),
    };
    let cfg = Config { time_ms, max_depth: 40, max_plies: 400, random_plies: 6, epsilon: 0.0, rules: Rules::Cycle };
    let opts = SearchOptions::parse("lmr2")?;
    println!("SPSA (새 규칙): 반복당 {}판, 수당 {}ms, 스레드 {}, 저장 {}", games, time_ms, threads, out.display());

    while iter < iters && !stop.should_stop() {
        let t0 = Instant::now();
        let d = signs(seed.wrapping_mul(1_000_003).wrapping_add(iter as u64), params.len());
        let shift = |sign: f64| -> Vec<f64> {
            theta.iter().zip(&params).zip(&d).map(|((v, p), s)| (v + sign * p.c * s).clamp(p.lo, p.hi)).collect()
        };
        let (plus, minus) = (shift(1.0), shift(-1.0));
        let pa = Player::new(weights_of(&plus), None).with_options(opts).with_tunables(Tunables::parse(&tune_str(&plus))?);
        let pb = Player::new(weights_of(&minus), None).with_options(opts).with_tunables(Tunables::parse(&tune_str(&minus))?);
        let (ab, ba) = ([pa.clone(), pb.clone()], [pb, pa]);
        let tally = Mutex::new((0i64, 0i64));
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let pair = next.fetch_add(1, Ordering::SeqCst);
                    if pair >= games / 2 || stop.should_stop() {
                        break;
                    }
                    let os = (seed as i64).wrapping_mul(7919).wrapping_add((iter * 1000 + pair) as i64);
                    let first = arena::play(&cfg, &ab, os, None, false, pair * 2);
                    let second = arena::play(&cfg, &ba, os, None, false, pair * 2 + 1);
                    let mut t = tally.lock().unwrap();
                    // A = plus 쪽
                    match first.winner {
                        0 => t.0 += 1,
                        1 => t.1 += 1,
                        _ => {}
                    }
                    match second.winner {
                        0 => t.1 += 1,
                        1 => t.0 += 1,
                        _ => {}
                    }
                });
            }
        });
        if stop.should_stop() {
            break;
        }
        let (aw, bw) = tally.into_inner().unwrap();
        let net = (aw - bw) as f64 / games as f64;
        for i in 0..theta.len() {
            theta[i] = (theta[i] + lr * params[i].c * net * d[i]).clamp(params[i].lo, params[i].hi);
        }
        iter += 1;
        save(&out, &state_path, iter, &theta)?;
        println!("[{}] {:.0}초, +쪽 {}승 -쪽 {}승 | {}", iter, t0.elapsed().as_secs_f64(), aw, bw, tune_str(&theta));
    }
    save(&out, &state_path, iter, &theta)?;
    println!("멈춤: 반복 {}. 지금 값 -> {}, {}", iter, out.join("spsa_tune.txt").display(), out.join("spsa_weights.json").display());
    Ok(())
}

fn save(out: &Path, state_path: &Path, iter: usize, theta: &[f64]) -> Result<(), String> {
    let text = format!("{}\n{}\n", iter, theta.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(","));
    let tmp = state_path.with_extension("tmp");
    fs::write(&tmp, text).and_then(|_| fs::rename(&tmp, state_path)).map_err(|e| e.to_string())?;
    fs::write(out.join("spsa_tune.txt"), tune_str(theta)).map_err(|e| e.to_string())?;
    fs::write(out.join("spsa_weights.json"), weights_of(theta).to_json()).map_err(|e| e.to_string())?;
    Ok(())
}

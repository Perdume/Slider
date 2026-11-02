//! 자가대전 학습 데이터 생성기 (2v2). 같은 가중치의 AI 끼리 여러 판을 두고,
//! AI 가 직접 고른 수 앞의 국면마다 한 줄(JSONL)씩 기록한다. Java `SelfPlay` 와 출력 형식이 같다.
//!
//! 한 줄: {"game":0,"ply":12,"side":0,"features":[...],"score":37,"outcome":1.0}
//!
//! 사용법:
//!   selfplay --games 200 --time-ms 30 --out data/selfplay.jsonl
//! 옵션: --games --time-ms --depth --random-plies --epsilon --max-plies --threads --seed --weights --out
//!       --net 신경망 파일 (있으면 평가에 더한다)
//!       --positions yes  각 줄에 국면(tiles, me, opp)도 기록한다 (신경망 학습용. 기본은 기록하지 않음)

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use std::sync::Arc;

use slider::arena::{self, Config, Player, Sample};
use slider::cli;
use slider::javarandom::JavaRandom;
use slider::nnue::Network;
use slider::weights::EvalWeights;

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let games: usize = cli::get(&opt, "games", 200)?;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads())?.max(1);
    let seed: i64 = cli::get(&opt, "seed", 1)?;
    let weights = EvalWeights::from_argument(opt.get("weights").map(String::as_str))?;
    let net = match opt.get("net") {
        Some(path) => Some(Arc::new(Network::load(Path::new(path))?)),
        None => None,
    };
    let positions_flag = opt.get("positions").map(|v| v == "yes").unwrap_or(false);
    let out = opt.get("out").cloned().unwrap_or_else(|| "selfplay.jsonl".to_string());

    let cfg = Config {
        time_ms: cli::get(&opt, "time-ms", 30)?,
        max_depth: cli::get(&opt, "depth", 40)?,
        random_plies: cli::get(&opt, "random-plies", 4)?,
        epsilon: cli::get(&opt, "epsilon", 0.05)?,
        max_plies: cli::get(&opt, "max-plies", 150)?,
    };

    if let Some(parent) = Path::new(&out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    println!(
        "자가대전 시작: games={}, time={}ms, threads={}, randomPlies={}, epsilon={:.2}, weights={}",
        games, cfg.time_ms, threads, cfg.random_plies, cfg.epsilon, weights
    );

    let file = File::create(&out).map_err(|e| format!("{}: {}", out, e))?;
    let writer = Mutex::new(BufWriter::new(file));
    let tally = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)]; // 먼저 둔 쪽 승, 무, 나중 쪽 승
    let positions = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let next_game = AtomicUsize::new(0);
    let start = Instant::now();
    let pair = [Player::new(weights.clone(), net.clone()), Player::new(weights.clone(), net.clone())];

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let game_id = next_game.fetch_add(1, Ordering::SeqCst);
                if game_id >= games {
                    break;
                }
                let game_seed = seed.wrapping_mul(1_000_003).wrapping_add(game_id as i64);
                let mut noise = JavaRandom::new(game_seed ^ 0x5DEE_CE66D);
                let outcome = arena::play(&cfg, &pair, game_seed, Some(&mut noise), true, game_id);
                let slot = if outcome.winner < 0 { 1 } else if outcome.winner == 0 { 0 } else { 2 };
                tally[slot].fetch_add(1, Ordering::SeqCst);
                positions.fetch_add(outcome.samples.len(), Ordering::SeqCst);
                let mut text = String::new();
                for s in &outcome.samples {
                    text.push_str(&to_json(s, positions_flag));
                    text.push('\n');
                }
                writer.lock().unwrap().write_all(text.as_bytes()).expect("쓰기 실패");
                let done = finished.fetch_add(1, Ordering::SeqCst) + 1;
                if done % 10 == 0 || done == games {
                    println!(
                        "  {}/{}판 (국면 {}개, {:.0}초)",
                        done,
                        games,
                        positions.load(Ordering::SeqCst),
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    writer.lock().unwrap().flush().map_err(|e| e.to_string())?;
    println!(
        "끝: 먼저 둔 쪽 {}승 / 무승부 {} / 나중 쪽 {}승, 국면 {}개 -> {}",
        tally[0].load(Ordering::SeqCst),
        tally[1].load(Ordering::SeqCst),
        tally[2].load(Ordering::SeqCst),
        positions.load(Ordering::SeqCst),
        out
    );
    Ok(())
}

fn to_json(s: &Sample, with_positions: bool) -> String {
    let features: Vec<String> = s.features.iter().map(|f| f.to_string()).collect();
    let position = if with_positions {
        format!(",\"tiles\":{},\"me\":{},\"opp\":{}", s.tiles, s.me, s.opp)
    } else {
        String::new()
    };
    format!(
        "{{\"game\":{},\"ply\":{},\"side\":{},\"features\":[{}],\"score\":{},\"outcome\":{:?}{}}}",
        s.game,
        s.ply,
        s.side,
        features.join(","),
        s.score,
        s.outcome,
        position
    )
}

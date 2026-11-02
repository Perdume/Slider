//! 가중치 두 벌(A, B)을 맞붙이는 벤치마크 (2v2). 시작 국면 하나마다 쪽을 바꿔 두 판씩 둬서 선공/후공 차이를 없앤다.
//! Java `Match` 와 출력 형식이 같다 (마지막 `RESULT {...}` 줄을 노트북이 읽는다).
//!
//! 사용법:
//!   matchup --a weights.json --b default --games 100 --time-ms 50
//! 옵션: --net-a --net-b 쪽별 신경망 파일 (없으면 선형 평가만)
//!       --stop-file 경로 / --parent-pid N  멈춤 신호 (멈추면 RESULT 줄 없이 끝난다)
//!       --a --b (JSON 경로 또는 default) --games (짝수로 올림) --time-ms --depth --random-plies --max-plies --threads --seed

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use slider::arena::{self, Config, Player};
use slider::cli;
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
    let a = EvalWeights::from_argument(opt.get("a").map(String::as_str))?;
    let b = EvalWeights::from_argument(opt.get("b").map(String::as_str))?;
    let load_net = |key: &str| -> Result<Option<Arc<Network>>, String> {
        match opt.get(key) {
            Some(path) => Ok(Some(Arc::new(Network::load(Path::new(path))?))),
            None => Ok(None),
        }
    };
    let net_a = load_net("net-a")?;
    let net_b = load_net("net-b")?;
    let stop = cli::Stop::from_options(&opt)?;
    let games: usize = cli::get(&opt, "games", 100)?;
    let pairs = (games + 1) / 2;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads())?.max(1);
    let seed: i64 = cli::get(&opt, "seed", 1)?;

    let cfg = Config {
        time_ms: cli::get(&opt, "time-ms", 50)?,
        max_depth: cli::get(&opt, "depth", 40)?,
        // 시작 국면을 다양하게 하려고 처음 몇 수는 무작위로 둔다 (한 쌍의 두 판은 같은 시작 국면)
        random_plies: cli::get(&opt, "random-plies", 6)?,
        max_plies: cli::get(&opt, "max-plies", 150)?,
        epsilon: 0.0,
    };

    println!("A = {}{}\nB = {}{}\n{}판 (시작 국면 {}개 x 쪽 교대), time={}ms, threads={}", a, if net_a.is_some() { " + 신경망" } else { "" }, b, if net_b.is_some() { " + 신경망" } else { "" }, pairs * 2, pairs, cfg.time_ms, threads);

    let a_wins = AtomicUsize::new(0);
    let b_wins = AtomicUsize::new(0);
    let draws = AtomicUsize::new(0);
    let next_pair = AtomicUsize::new(0);
    let pa = Player::new(a.clone(), net_a);
    let pb = Player::new(b.clone(), net_b);
    let ab = [pa.clone(), pb.clone()];
    let ba = [pb, pa];

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                if stop.should_stop() {
                    break;
                }
                let pair = next_pair.fetch_add(1, Ordering::SeqCst);
                if pair >= pairs {
                    break;
                }
                let opening_seed = seed.wrapping_mul(7919).wrapping_add(pair as i64);
                // 첫 판: A 가 먼저 두는 쪽, 둘째 판: B 가 먼저 두는 쪽
                let first = arena::play(&cfg, &ab, opening_seed, None, false, pair * 2);
                let second = arena::play(&cfg, &ba, opening_seed, None, false, pair * 2 + 1);
                count(first.winner, true, &a_wins, &b_wins, &draws);
                count(second.winner, false, &a_wins, &b_wins, &draws);
            });
        }
    });

    if stop.should_stop() {
        println!("중단됨");
        return Ok(());
    }
    let games = pairs * 2;
    let (aw, dw, bw) = (a_wins.load(Ordering::SeqCst), draws.load(Ordering::SeqCst), b_wins.load(Ordering::SeqCst));
    let score = (aw as f64 + 0.5 * dw as f64) / games as f64;
    // 판마다 결과가 1/0.5/0 이므로 표본 분산으로 대략적인 95% 구간을 낸다
    let var = (aw as f64 * sq(1.0 - score) + dw as f64 * sq(0.5 - score) + bw as f64 * sq(score)) / games as f64;
    let margin = 1.96 * (var / games as f64).sqrt();
    println!(
        "A {}승 / 무 {} / B {}승  ->  A 점수 {:.3} (95% 구간 약 {:.3} ~ {:.3})",
        aw, dw, bw, score, score - margin, score + margin
    );
    // 노트북 등 다른 프로그램이 읽기 쉬운 한 줄
    println!(
        "RESULT {{\"aWins\":{},\"draws\":{},\"bWins\":{},\"score\":{:.4},\"low\":{:.4},\"high\":{:.4}}}",
        aw, dw, bw, score, score - margin, score + margin
    );
    Ok(())
}

// 먼저 둔 쪽이 이기면 winner 0. a_first 면 먼저 둔 쪽이 A.
fn count(winner: i32, a_first: bool, a_wins: &AtomicUsize, b_wins: &AtomicUsize, draws: &AtomicUsize) {
    if winner < 0 {
        draws.fetch_add(1, Ordering::SeqCst);
    } else if (winner == 0) == a_first {
        a_wins.fetch_add(1, Ordering::SeqCst);
    } else {
        b_wins.fetch_add(1, Ordering::SeqCst);
    }
}

fn sq(x: f64) -> f64 {
    x * x
}

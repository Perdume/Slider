//! 가중치 두 벌(A, B)을 맞붙이는 벤치마크 (2v2). 시작 국면 하나마다 쪽을 바꿔 두 판씩 둬서 선공/후공 차이를 없앤다.
//! Java `Match` 와 출력 형식이 같다 (마지막 `RESULT {...}` 줄을 노트북이 읽는다).
//!
//! 사용법:
//!   matchup --a weights.json --b default --games 100 --time-ms 50
//! 옵션: --net-a --net-b 쪽별 신경망 파일 (없으면 선형 평가만)
//!       --opts-a --opts-b 쪽별 탐색 개선 선택지 (쉼표 목록: pvs,lmr,lmr2,nmp. 기본은 모두 끔)
//!       --time-ms-a --time-ms-b 쪽별 수당 시간 (없으면 --time-ms)
//!       --stop-file 경로 / --parent-pid N  멈춤 신호 (멈추면 RESULT 줄 없이 끝난다)
//!       --a --b (JSON 경로 또는 default) --games (짝수로 올림) --time-ms --depth --random-plies --max-plies --threads --seed
//!       --sprt-elo0 0 --sprt-elo1 20 [--sprt-alpha 0.05 --sprt-beta 0.05]
//!           SPRT 로 일찍 끝낸다: "A 가 B 보다 elo1 만큼 강하다(H1)" 와 "elo0 이하다(H0)" 중 하나가 통계적으로 정해지면 멈춘다.
//!           --games 는 최대 판 수가 된다. 무승부가 많아도 판 수를 알아서 늘리거나 줄인다.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use slider::ai::SearchOptions;
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
    let opts_a = SearchOptions::parse(opt.get("opts-a").map(String::as_str).unwrap_or(""))?;
    let opts_b = SearchOptions::parse(opt.get("opts-b").map(String::as_str).unwrap_or(""))?;
    let side_time = |key: &str| -> Result<Option<u64>, String> {
        match opt.get(key) {
            Some(v) => v.parse::<u64>().map(Some).map_err(|_| format!("--{} 는 숫자여야 합니다: {}", key, v)),
            None => Ok(None),
        }
    };
    let (time_a, time_b) = (side_time("time-ms-a")?, side_time("time-ms-b")?);
    if time_a.is_some() || time_b.is_some() {
        println!("수당 시간: A {}ms / B {}ms", time_a.unwrap_or(0), time_b.unwrap_or(0));
    }
    let stop = cli::Stop::from_options(&opt)?;
    let games: usize = cli::get(&opt, "games", 100)?;
    let pairs = (games + 1) / 2;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads())?.max(1);
    let seed: i64 = cli::get(&opt, "seed", 1)?;
    let sprt = match (opt.get("sprt-elo0"), opt.get("sprt-elo1")) {
        (None, None) => None,
        (Some(_), Some(_)) => {
            let (e0, e1): (f64, f64) = (cli::get(&opt, "sprt-elo0", 0.0)?, cli::get(&opt, "sprt-elo1", 20.0)?);
            let (alpha, beta): (f64, f64) = (cli::get(&opt, "sprt-alpha", 0.05)?, cli::get(&opt, "sprt-beta", 0.05)?);
            if e1 <= e0 {
                return Err("--sprt-elo1 은 --sprt-elo0 보다 커야 합니다".into());
            }
            Some(Sprt { elo0: e0, elo1: e1, lower: (beta / (1.0 - alpha)).ln(), upper: ((1.0 - beta) / alpha).ln() })
        }
        _ => return Err("--sprt-elo0 과 --sprt-elo1 은 같이 줘야 합니다".into()),
    };

    let cfg = Config {
        time_ms: cli::get(&opt, "time-ms", 50)?,
        max_depth: cli::get(&opt, "depth", 40)?,
        // 시작 국면을 다양하게 하려고 처음 몇 수는 무작위로 둔다 (한 쌍의 두 판은 같은 시작 국면)
        random_plies: cli::get(&opt, "random-plies", 6)?,
        max_plies: cli::get(&opt, "max-plies", 150)?,
        epsilon: 0.0,
    };

    println!("A = {}{}\nB = {}{}\n{}판{} (시작 국면 {}개 x 쪽 교대), time={}ms, depth={}, threads={}", a, if net_a.is_some() { " + 신경망" } else { "" }, b, if net_b.is_some() { " + 신경망" } else { "" }, pairs * 2, if sprt.is_some() { " 이하" } else { "" }, pairs, cfg.time_ms, cfg.max_depth, threads);
    if let Some(sp) = &sprt {
        println!("SPRT: H0 = Elo {} 이하, H1 = Elo {} 이상, LLR 경계 [{:.2}, {:.2}]", sp.elo0, sp.elo1, sp.lower, sp.upper);
    }

    // (A 승, 무, B 승)
    let tally = Mutex::new((0usize, 0usize, 0usize));
    let decided = AtomicBool::new(false);
    // 경계를 넘은 순간의 판정 (그 뒤에 이미 두고 있던 판이 끝나서 LLR 이 경계 안으로 돌아와도 판정은 그대로)
    let verdict_at_cross: Mutex<Option<&'static str>> = Mutex::new(None);
    let next_pair = AtomicUsize::new(0);
    if opts_a != SearchOptions::default() || opts_b != SearchOptions::default() {
        println!("탐색: A {} / B {}", opts_a, opts_b);
    }
    let pa = Player::new(a.clone(), net_a).with_options(opts_a).with_time(time_a);
    let pb = Player::new(b.clone(), net_b).with_options(opts_b).with_time(time_b);
    let ab = [pa.clone(), pb.clone()];
    let ba = [pb, pa];

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                if stop.should_stop() || decided.load(Ordering::SeqCst) {
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
                let mut t = tally.lock().unwrap();
                count(first.winner, true, &mut t);
                count(second.winner, false, &mut t);
                if let Some(sp) = &sprt {
                    let llr = sp.llr(t.0, t.1, t.2);
                    let games = t.0 + t.1 + t.2;
                    if games % 40 == 0 {
                        println!("  {}판: A {}승 / 무 {} / B {}승, LLR {:.2} [{:.2}, {:.2}]", games, t.0, t.1, t.2, llr, sp.lower, sp.upper);
                    }
                    if (llr >= sp.upper || llr <= sp.lower) && !decided.swap(true, Ordering::SeqCst) {
                        *verdict_at_cross.lock().unwrap() = Some(if llr >= sp.upper { "H1" } else { "H0" });
                    }
                }
            });
        }
    });

    if stop.should_stop() {
        println!("중단됨");
        return Ok(());
    }
    let (aw, dw, bw) = *tally.lock().unwrap();
    let games = aw + dw + bw;
    if games == 0 {
        return Err("둔 판이 없습니다".into());
    }
    let score = (aw as f64 + 0.5 * dw as f64) / games as f64;
    // 판마다 결과가 1/0.5/0 이므로 표본 분산으로 대략적인 95% 구간을 낸다
    let var = (aw as f64 * sq(1.0 - score) + dw as f64 * sq(0.5 - score) + bw as f64 * sq(score)) / games as f64;
    let margin = 1.96 * (var / games as f64).sqrt();
    let (elo, elo_low, elo_high) = (elo_of(score), elo_of(score - margin), elo_of(score + margin));
    println!(
        "A {}승 / 무 {} / B {}승 ({}판)  ->  A 점수 {:.3} (95% 구간 약 {:.3} ~ {:.3}), Elo {:+.0} ({:+.0} ~ {:+.0})",
        aw, dw, bw, games, score, score - margin, score + margin, elo, elo_low, elo_high
    );
    let mut sprt_json = String::new();
    if let Some(sp) = &sprt {
        let llr = sp.llr(aw, dw, bw);
        let at_end = if llr >= sp.upper { "H1" } else if llr <= sp.lower { "H0" } else { "undecided" };
        let verdict = verdict_at_cross.lock().unwrap().unwrap_or(at_end);
        let text = match verdict {
            "H1" => format!("A 가 B 보다 강하다 (Elo {} 이상 쪽 채택)", sp.elo1),
            "H0" => format!("A 가 B 보다 Elo {} 만큼 강하지는 않다 (H0 채택)", sp.elo1),
            _ => "최대 판 수까지 결론이 안 남".to_string(),
        };
        println!("SPRT 결과: {} — LLR {:.2} [{:.2}, {:.2}]", text, llr, sp.lower, sp.upper);
        sprt_json = format!(
            ",\"sprt\":\"{}\",\"llr\":{:.3},\"lower\":{:.3},\"upper\":{:.3},\"elo0\":{},\"elo1\":{}",
            verdict, llr, sp.lower, sp.upper, sp.elo0, sp.elo1
        );
    }
    // 노트북 등 다른 프로그램이 읽기 쉬운 한 줄
    println!(
        "RESULT {{\"aWins\":{},\"draws\":{},\"bWins\":{},\"games\":{},\"score\":{:.4},\"low\":{:.4},\"high\":{:.4},\"elo\":{:.1},\"eloLow\":{:.1},\"eloHigh\":{:.1}{}}}",
        aw, dw, bw, games, score, score - margin, score + margin, elo, elo_low, elo_high, sprt_json
    );
    Ok(())
}

// 먼저 둔 쪽이 이기면 winner 0. a_first 면 먼저 둔 쪽이 A. t = (A 승, 무, B 승)
fn count(winner: i32, a_first: bool, t: &mut (usize, usize, usize)) {
    if winner < 0 {
        t.1 += 1;
    } else if (winner == 0) == a_first {
        t.0 += 1;
    } else {
        t.2 += 1;
    }
}

/// 점수(0..1) -> Elo 차이 (로지스틱). 끝값은 잘라서 무한대를 피한다.
fn elo_of(score: f64) -> f64 {
    let s = score.clamp(0.001, 0.999);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// 승/무/패 3항 SPRT (일반화 SPRT 의 정규 근사, fishtest 와 같은 꼴).
/// 한 쌍의 두 판은 같은 시작 국면이라 서로 상관이 있지만 3항 모형은 이를 무시한다 (보통 판 수를 조금 더 쓰는 쪽으로 보수적).
struct Sprt {
    elo0: f64,
    elo1: f64,
    lower: f64,
    upper: f64,
}

impl Sprt {
    fn llr(&self, w: usize, d: usize, l: usize) -> f64 {
        let n = (w + d + l) as f64;
        if n == 0.0 {
            return 0.0;
        }
        let s = (w as f64 + 0.5 * d as f64) / n;
        let var = (w as f64 * sq(1.0 - s) + d as f64 * sq(0.5 - s) + l as f64 * sq(s)) / n;
        if var <= 0.0 {
            return 0.0;
        }
        let expected = |elo: f64| 1.0 / (1.0 + 10f64.powf(-elo / 400.0));
        let (s0, s1) = (expected(self.elo0), expected(self.elo1));
        (s1 - s0) * (2.0 * s - s0 - s1) / (2.0 * var / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sprt(e0: f64, e1: f64) -> Sprt {
        Sprt { elo0: e0, elo1: e1, lower: (0.05f64 / 0.95).ln(), upper: (0.95f64 / 0.05).ln() }
    }

    #[test]
    fn llr_points_the_right_way() {
        let sp = sprt(0.0, 20.0);
        // 확실히 강하면 위로, 확실히 약하면 아래로, 반반이면 그 사이(약간 아래)
        assert!(sp.llr(400, 400, 200) > sp.upper);
        assert!(sp.llr(200, 400, 400) < sp.lower);
        let even = sp.llr(300, 400, 300);
        assert!(even < 0.0 && even > sp.lower, "{}", even);
        assert_eq!(sp.llr(0, 10, 0), 0.0);
    }

    #[test]
    fn elo_of_is_symmetric_and_zero_at_half() {
        assert!(elo_of(0.5).abs() < 1e-9);
        assert!((elo_of(0.64) + elo_of(0.36)).abs() < 1e-9);
        assert!((elo_of(0.64) - 100.0).abs() < 1.0);
    }
}

fn sq(x: f64) -> f64 {
    x * x
}

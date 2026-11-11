//! 학습 루프 전체를 한 프로세스로 돌린다: 자가대전 -> 신경망 학습 -> 대국 확인 -> 채택, 이를 라운드만큼 반복.
//! Colab 노트북(colab/slider_nnue_training.ipynb)과 같은 흐름이고, 안드로이드 앱이 이 파일 하나만 실행한다.
//! selfplay / trainnet / matchup 실행 파일을 자식 프로세스로 부르며, 출력은 들여써서 그대로 보여 준다.
//!
//! 사용법:
//!   trainloop --work-dir work --rounds 2 --games 600 --time-ms 30 --epochs 40 --match-games 200
//! 옵션: --work-dir (필수) --selfplay-bin --trainnet-bin --matchup-bin (기본: 이 파일과 같은 폴더의 selfplay, trainnet, matchup)
//!       --rounds 2 --games 600 --time-ms 30 --random-plies 4 --epsilon 0.05 --epochs 40 --lambda 1.0
//!       --match-games 200 --match-time-ms 50 --accept 0.55 --weights default --net (이어서 학습할 신경망) --threads --seed 1
//!       --stop-file 경로 / --parent-pid N  멈춤 신호 (cli::Stop 참고)
//! 결과: <work-dir>/net.txt (채택된 신경망이 있을 때) 와 <work-dir>/summary.txt

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use slider::cli;

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

struct Tools {
    selfplay: PathBuf,
    trainnet: PathBuf,
    matchup: PathBuf,
}

/// 자식 프로세스를 돌리고 출력 줄을 실시간으로 보여준 뒤 모아서 돌려준다.
fn run_child(bin: &Path, args: &[String]) -> Result<Vec<String>, String> {
    println!("$ {} {}", bin.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), args.join(" "));
    let mut child = Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("{} 를 실행할 수 없습니다: {}", bin.display(), e))?;
    let mut lines = Vec::new();
    for line in BufReader::new(child.stdout.take().expect("piped")).lines() {
        let line = line.map_err(|e| e.to_string())?;
        println!("    {}", line);
        lines.push(line);
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("{} 실행 실패 ({})", bin.display(), status));
    }
    Ok(lines)
}

/// `RESULT {"aWins":..,"draws":..,"bWins":..,"score":..}` 줄에서 숫자 하나를 읽는다.
fn result_field(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\":", key);
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let work = PathBuf::from(opt.get("work-dir").ok_or("--work-dir 가 필요합니다")?);
    let stop = cli::Stop::from_options(&opt)?;
    let rounds: usize = cli::get(&opt, "rounds", 2)?;
    let games: usize = cli::get(&opt, "games", 600)?;
    let time_ms: u64 = cli::get(&opt, "time-ms", 30)?;
    let random_plies: usize = cli::get(&opt, "random-plies", 4)?;
    let epsilon: f64 = cli::get(&opt, "epsilon", 0.05)?;
    let epochs: usize = cli::get(&opt, "epochs", 40)?;
    let lambda: f64 = cli::get(&opt, "lambda", 1.0)?;
    let match_games: usize = cli::get(&opt, "match-games", 200)?;
    let match_time: u64 = cli::get(&opt, "match-time-ms", 50)?;
    let accept: f64 = cli::get(&opt, "accept", 0.55)?;
    let seed: i64 = cli::get(&opt, "seed", 1)?;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads())?.max(1);
    let weights = opt.get("weights").cloned().unwrap_or_else(|| "default".to_string());

    let sibling = |name: &str| -> PathBuf {
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join(name))).unwrap_or_else(|| PathBuf::from(name))
    };
    let tools = Tools {
        selfplay: opt.get("selfplay-bin").map(PathBuf::from).unwrap_or_else(|| sibling("selfplay")),
        trainnet: opt.get("trainnet-bin").map(PathBuf::from).unwrap_or_else(|| sibling("trainnet")),
        matchup: opt.get("matchup-bin").map(PathBuf::from).unwrap_or_else(|| sibling("matchup")),
    };

    let data_dir = work.join("data");
    std::fs::create_dir_all(&data_dir).map_err(|e| format!("{}: {}", data_dir.display(), e))?;
    // 자식에게 넘길 멈춤 신호: 이 프로세스가 사라지면 자식도 멈추게 한다 (리눅스/안드로이드)
    let mut stop_args = stop.child_args();
    if cfg!(any(target_os = "linux", target_os = "android")) && !stop_args.iter().any(|a| a == "--parent-pid") {
        stop_args.push("--parent-pid".to_string());
        stop_args.push(std::process::id().to_string());
    }

    let mut current_net: Option<PathBuf> = opt.get("net").map(PathBuf::from);
    let mut data_files: Vec<PathBuf> = Vec::new();
    let mut summary: Vec<String> = Vec::new();
    println!(
        "학습 시작: rounds={}, games/round={}, time={}ms, threads={}, 선형 가중치={}, 시작 신경망={}",
        rounds,
        games,
        time_ms,
        threads,
        weights,
        current_net.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "없음".to_string())
    );

    for round in 1..=rounds {
        if stop.should_stop() {
            break;
        }
        println!("\n=============== 라운드 {}/{} ===============", round, rounds);
        println!("[1/3] 자가대전 {}판 (현재 평가: 선형 + {})", games, if current_net.is_some() { "신경망" } else { "신경망 없음" });
        let data = data_dir.join(format!("selfplay_round{}.jsonl", round));
        let mut a = vec![
            "--games".into(), games.to_string(), "--time-ms".into(), time_ms.to_string(),
            "--random-plies".into(), random_plies.to_string(), "--epsilon".into(), epsilon.to_string(),
            "--threads".into(), threads.to_string(), "--seed".into(), (seed + round as i64).to_string(),
            "--weights".into(), weights.clone(), "--positions".into(), "yes".into(),
            "--out".into(), data.display().to_string(),
        ];
        if let Some(n) = &current_net {
            a.push("--net".into());
            a.push(n.display().to_string());
        }
        a.extend(stop_args.iter().cloned());
        run_child(&tools.selfplay, &a)?;
        if stop.should_stop() {
            break;
        }
        data_files.push(data);

        println!("[2/3] 신경망 학습 (데이터 {}개 파일 전체)", data_files.len());
        let candidate = work.join(format!("net_round{}.txt", round));
        let mut a = vec![
            "--data".into(), data_files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(","),
            "--weights".into(), weights.clone(), "--out".into(), candidate.display().to_string(),
            "--epochs".into(), epochs.to_string(), "--lambda".into(), lambda.to_string(), "--seed".into(), seed.to_string(),
        ];
        a.extend(stop_args.iter().cloned());
        run_child(&tools.trainnet, &a)?;
        if stop.should_stop() {
            break;
        }

        println!("[3/3] 대국 확인: 새 신경망 vs {}", if current_net.is_some() { "지금까지 채택한 평가" } else { "선형 평가만" });
        let mut a = vec![
            "--a".into(), weights.clone(), "--net-a".into(), candidate.display().to_string(),
            "--b".into(), weights.clone(), "--games".into(), match_games.to_string(),
            "--time-ms".into(), match_time.to_string(), "--threads".into(), threads.to_string(),
            "--seed".into(), (seed + 1000).to_string(),
        ];
        if let Some(n) = &current_net {
            a.push("--net-b".into());
            a.push(n.display().to_string());
        }
        a.extend(stop_args.iter().cloned());
        let lines = run_child(&tools.matchup, &a)?;
        if stop.should_stop() {
            break;
        }
        let line = lines.iter().rev().find(|l| l.starts_with("RESULT ")).ok_or("대국 결과(RESULT) 줄이 없습니다")?;
        let get = |k: &str| result_field(line, k).ok_or_else(|| format!("RESULT 에서 {} 를 읽을 수 없습니다: {}", k, line));
        let (wins, draws, losses, score) = (get("aWins")?, get("draws")?, get("bWins")?, get("score")?);
        let accepted = score >= accept;
        let text = format!(
            "라운드 {}: 새 신경망 {}승 {}무 {}패, 점수 {:.3} => {}",
            round, wins, draws, losses, score, if accepted { "채택" } else { "채택 안 함" }
        );
        println!("    -> {}", text);
        summary.push(text);
        if accepted {
            current_net = Some(candidate);
        }
    }

    let stopped = stop.should_stop();
    println!("\n{}", if stopped { "중단됨" } else { "라운드 요약" });
    for l in &summary {
        println!("  {}", l);
    }
    let final_net = work.join("net.txt");
    match &current_net {
        Some(n) if Some(n) != opt.get("net").map(PathBuf::from).as_ref() || summary.is_empty() => {
            std::fs::copy(n, &final_net).map_err(|e| e.to_string())?;
            println!("저장: {}", final_net.display());
        }
        Some(_) => println!("새로 채택된 신경망이 없습니다 (시작 신경망 그대로)."),
        None => println!("채택된 신경망이 없습니다. 판 수(--games)나 수당 시간(--time-ms)을 늘려서 다시 돌려 보세요."),
    }
    std::fs::write(work.join("summary.txt"), summary.join("\n") + "\n").map_err(|e| e.to_string())?;
    Ok(())
}

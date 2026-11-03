//! 다른 프로그램(파이썬 강화학습 등)이 Rust 엔진의 수를 받아 가는 서버.
//! 표준 입력으로 국면을 한 줄에 하나씩 받고, 빈 줄이 오면 그때까지 받은 국면들을 병렬로 탐색해서 같은 순서로 답한다.
//!
//!   입력 한 줄:  tiles me opp [mc oc]   (u64 비트보드, me = 둘 차례인 쪽 말. tiles 에 말이 선 칸이 없어도 됨)
//!                mc oc 를 주면 순환 규칙: mc = 지금 움직일 내 말, oc = 상대가 다음에 움직일 말 (없으면 예전 자유 규칙)
//!   출력 한 줄:  move score depth        (move 는 엔진의 수 정수, 둘 수 없으면 -1 = 통과)
//!
//! 옵션: --time-ms 50 --weights 경로|default|tuned(기본) --search lmr2,hm(기본)|lmr2|none --tune game(기본)|default|이름=값,...
//!       --threads 동시에 탐색할 국면 수(기본 코어 수)
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use slider::ai::{SearchOptions, SliderAi, Tunables};
use slider::cli;
use slider::weights::EvalWeights;

const TUNED: &str = include_str!("../../../weights/tuned.json");

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let time_ms: u64 = cli::get(&opt, "time-ms", 50)?;
    let threads: usize = cli::get(&opt, "threads", cli::default_threads().max(1))?.max(1);
    let opts = SearchOptions::parse(opt.get("search").map(String::as_str).unwrap_or("lmr2,hm"))?;
    // 탐색 상수: 기본은 SPSA 로 맞춘 게임용 값. --tune 으로 바꿈 (예: --tune margin=40,trap=60 / 예전 값은 --tune default)
    let tune = match opt.get("tune").map(String::as_str) {
        None | Some("game") => Tunables::game(),
        Some("default") => Tunables::default(),
        Some(t) => Tunables::parse(t)?,
    };
    let weights = match opt.get("weights").map(String::as_str) {
        None | Some("tuned") => EvalWeights::from_json(TUNED)?,
        Some(other) => EvalWeights::from_argument(Some(other))?,
    };
    let stdin = io::stdin();
    let mut batch: Vec<(u64, u64, u64, u64, u64)> = Vec::new();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let t = line.trim();
        if !t.is_empty() {
            let v: Vec<u64> = t.split_whitespace().map(|x| x.parse::<u64>()).collect::<Result<_, _>>().map_err(|_| format!("읽을 수 없는 줄: {}", t))?;
            match v.len() {
                3 => batch.push((v[0], v[1], v[2], v[1], v[2])),
                5 => batch.push((v[0], v[1], v[2], v[3], v[4])),
                _ => return Err(format!("tiles me opp [mc oc] 가 필요합니다: {}", t)),
            }
            continue;
        }
        // 빈 줄: 모아 둔 국면을 병렬로 탐색
        let results = Mutex::new(vec![(0i32, 0i32, 0i32); batch.len()]);
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads.min(batch.len().max(1)) {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= batch.len() {
                        break;
                    }
                    let (tiles, me, opp, mc, oc) = batch[i];
                    let mut ai = SliderAi::with_weights(time_ms, 40, &weights);
                    ai.set_options(opts);
                    ai.set_tunables(tune);
                    let r = ai.search_turn(tiles, me, opp, mc, oc);
                    results.lock().unwrap()[i] = (r.mv, r.score, r.depth);
                });
            }
        });
        for (mv, score, depth) in results.into_inner().unwrap() {
            writeln!(out, "{} {} {}", mv, score, depth).map_err(|e| e.to_string())?;
        }
        writeln!(out).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        batch.clear();
    }
    Ok(())
}

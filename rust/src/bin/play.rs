//! 콘솔에서 사람과 AI 가 두는 Slider.
//!
//!   cargo run --release --bin play
//! 규칙 (기본 --rules cycle): 대각선 배치 (왼쪽 위 AI, 오른쪽 위 나, 왼쪽 아래 나, 오른쪽 아래 AI).
//!   차례는 나부터 시계 방향으로 말 하나씩: 오른쪽 위(나 1) -> 오른쪽 아래(AI X) -> 왼쪽 아래(나 2) -> 왼쪽 위(AI Y) -> ...
//!   --rules free 는 예전 규칙 (위쪽 두 귀퉁이 나, 아래쪽 AI, 차례마다 말 아무거나).
//! 옵션: --time-ms 1500 (AI 수당 시간) --threads N (AI 가 한 수에 쓰는 스레드, 기본 코어 수)
//!       --search lmr2,hm|lmr2|none (탐색 개선, 기본 lmr2,hm) --tune game|default|이름=값,... (탐색 상수, 기본 game = SPSA 로 맞춘 값)
//!       --weights 경로|default|tuned (기본 tuned) --net 신경망 파일
//!
//! 조작: (free 규칙이면 움직일 말 1 또는 2 를 고른 뒤) w(위) a(왼쪽) s(아래) d(오른쪽).
//!   옆 칸이 타일이면 걸어가고 턴이 계속된다. 옆 칸이 빈 칸이면 서 있는 타일째 미끄러지고 턴이 끝난다.
//!   r = 이번 턴을 말 고르기부터 다시. q = 끝내기.
//! 어느 말이든 골(G, 가운데)에 멈추면 이긴다. 둘 수 있는 수가 없으면 차례가 넘어가고, 같은 국면이 3번 나오면 무승부다.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use slider::ai::{turn_hash, History, SearchOptions, SliderAi, Tunables, WIN_BOUND};
use slider::arena::{Rules, CYCLE_SIDE0_START, CYCLE_SIDE1_START, SIDE0_START, SIDE1_START, START_BOARD};
use slider::bitboard::{self as bb, GOAL, SIZE};
use slider::cli;
use slider::nnue::Network;
use slider::weights::EvalWeights;

/// 레포의 weights/tuned.json (Colab 에서 튜닝한 가중치). 기본 가중치보다 강했다.
const TUNED: &str = include_str!("../../../weights/tuned.json");
/// 예전 규칙(--rules free)용 가중치
const TUNED_FREE: &str = include_str!("../../../weights/tuned_free.json");
const PLAYER_MARK: [char; 2] = ['1', '2'];
const AI_MARK: [char; 2] = ['X', 'Y'];
// w a s d -> bitboard 방향 번호 (0 위, 1 아래, 2 왼쪽, 3 오른쪽)
const DIRS: [(&str, usize); 4] = [("w", 0), ("s", 1), ("a", 2), ("d", 3)];

struct State {
    tiles: u64,
    player: [usize; 2],
    ai: [usize; 2],
    /// 순환 규칙에서 다음에 움직일 말 번호 (플레이어, AI). 자유 규칙이면 None
    turn: Option<[usize; 2]>,
}

impl State {
    fn mask(cells: &[usize; 2]) -> u64 {
        (1u64 << cells[0]) | (1u64 << cells[1])
    }

    /// (플레이어 말 중 다음에 움직일 수 있는 말, AI 말 중 다음에 움직일 수 있는 말)
    fn movers(&self) -> (u64, u64) {
        match self.turn {
            Some([p, a]) => (1u64 << self.player[p], 1u64 << self.ai[a]),
            None => (State::mask(&self.player), State::mask(&self.ai)),
        }
    }

    /// 순환 규칙: 방금 둔(또는 넘긴) 쪽의 다음 말로 넘긴다
    fn advance(&mut self, player_moved: bool) {
        if let Some(t) = self.turn.as_mut() {
            let i = if player_moved { 0 } else { 1 };
            t[i] ^= 1;
        }
    }

    fn print(&self) {
        let mut out = String::new();
        for r in 0..SIZE {
            for c in 0..SIZE {
                let cell = r * SIZE + c;
                let mark = if let Some(i) = self.player.iter().position(|&p| p == cell) {
                    PLAYER_MARK[i]
                } else if let Some(i) = self.ai.iter().position(|&p| p == cell) {
                    AI_MARK[i]
                } else if bb::is_tile(self.tiles, cell) {
                    '■'
                } else if cell == GOAL {
                    'G'
                } else {
                    '□'
                };
                out.push(mark);
                out.push(' ');
            }
            out.push('\n');
        }
        let loc = |c: usize| format!("[{}, {}]", c / SIZE, c % SIZE);
        out.push_str(&format!(
            "플레이어 말: 1={} 2={}  AI 말: X={} Y={}",
            loc(self.player[0]),
            loc(self.player[1]),
            loc(self.ai[0]),
            loc(self.ai[1])
        ));
        if let Some([p, a]) = self.turn {
            out.push_str(&format!("\n다음 순서: 나 {} -> AI {}", PLAYER_MARK[p], AI_MARK[a]));
        }
        println!("{}", out);
    }

    /// 말 하나를 한 칸 방향으로 움직인다 (사람 조작). None = 갈 수 없음,
    /// Some(false) = 타일로 걸어감(턴 계속), Some(true) = 미끄러짐(턴 끝).
    fn step(&mut self, mine_is_player: bool, idx: usize, dir: usize) -> Option<bool> {
        let (mine, theirs) = if mine_is_player { (self.player, self.ai) } else { (self.ai, self.player) };
        let cur = mine[idx];
        let n = bb::neighbor(cur, dir);
        if n < 0 {
            return None;
        }
        let n = n as usize;
        if n == mine[1 - idx] || theirs.contains(&n) {
            return None;
        }
        let walked = bb::is_tile(self.tiles, n);
        let new_cell = if walked {
            n
        } else {
            // 빈 칸이면 서 있는 타일째 다른 타일이나 벽 앞까지 미끄러진다
            let mut dest = n;
            loop {
                let next = bb::neighbor(dest, dir);
                if next >= 0 && !bb::is_tile(self.tiles, next as usize) {
                    dest = next as usize;
                } else {
                    break;
                }
            }
            self.tiles = (self.tiles & !(1u64 << cur)) | (1u64 << dest);
            dest
        };
        if mine_is_player {
            self.player[idx] = new_cell;
        } else {
            self.ai[idx] = new_cell;
        }
        Some(!walked)
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let time_ms: u64 = cli::get(&opt, "time-ms", 1500)?;
    let threads: usize = cli::get(&opt, "threads", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))?;
    let opts = SearchOptions::parse(opt.get("search").map(String::as_str).unwrap_or("lmr2,hm"))?;
    // 탐색 상수: 기본은 SPSA 로 맞춘 게임용 값. --tune 으로 바꿈 (예: --tune margin=40,trap=60 / 예전 값은 --tune default)
    let rules = Rules::parse(opt.get("rules").map(String::as_str).unwrap_or("cycle"))?;
    let tune = match opt.get("tune").map(String::as_str) {
        None if rules == Rules::Free => Tunables::game_free(),
        None | Some("game") => Tunables::game(),
        Some("default") => Tunables::default(),
        Some(t) => Tunables::parse(t)?,
    };
    let weights = match opt.get("weights").map(String::as_str) {
        None if rules == Rules::Free => EvalWeights::from_json(TUNED_FREE)?,
        None | Some("tuned") => EvalWeights::from_json(TUNED)?,
        Some(other) => EvalWeights::from_argument(Some(other))?,
    };
    let net = match opt.get("net") {
        Some(p) => Some(Arc::new(Network::load(Path::new(p))?)),
        None => None,
    };
    println!("AI: 수당 {}ms, 스레드 {}, 탐색 {}{}{}", time_ms, threads, opts, if tune == Tunables::game() || tune == Tunables::game_free() { " + 튜닝 상수" } else { "" }, if net.is_some() { ", 신경망" } else { "" });

    let start = |locs: &[(usize, usize); 2]| [locs[0].0 * SIZE + locs[0].1, locs[1].0 * SIZE + locs[1].1];
    let mut st = match rules {
        Rules::Cycle => State { tiles: bb::from_array(&START_BOARD), player: start(&CYCLE_SIDE0_START), ai: start(&CYCLE_SIDE1_START), turn: Some([0, 0]) },
        Rules::Free => State { tiles: bb::from_array(&START_BOARD), player: start(&SIDE0_START), ai: start(&SIDE1_START), turn: None },
    };
    println!("규칙: {}", if rules == Rules::Cycle { "말 하나씩 시계 방향 (나 1 -> AI X -> 나 2 -> AI Y)" } else { "예전 규칙 (말 아무거나)" });
    let mut history = History::default();
    let mut player_to_move = true;
    record(&mut history, &st, player_to_move);
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        let outcome = if player_to_move { player_turn(&mut st, &mut lines)? } else { ai_turn(&mut st, &history, time_ms, threads, opts, tune, &weights, &net) };
        match outcome {
            Outcome::Quit => {
                println!("게임을 종료합니다.");
                return Ok(());
            }
            Outcome::Won => {
                st.print();
                println!("{}", if player_to_move { "You win" } else { "You lose" });
                return Ok(());
            }
            _ => {}
        }
        st.advance(player_to_move);
        player_to_move = !player_to_move;
        if record(&mut history, &st, player_to_move) >= 3 {
            st.print();
            println!("같은 국면이 3번 반복되어 무승부입니다.");
            return Ok(());
        }
    }
}

enum Outcome {
    Moved,
    Passed,
    Won,
    Quit,
}

/// 이 국면을 기록하고 지금까지 나온 횟수(이번 포함)를 돌려준다. 키는 둘 차례인 쪽 기준 position_hash.
fn record(history: &mut History, st: &State, player_to_move: bool) -> u32 {
    let (p, a) = (State::mask(&st.player), State::mask(&st.ai));
    let (pc, ac) = st.movers();
    let key = if player_to_move { turn_hash(st.tiles, p, a, pc, ac) } else { turn_hash(st.tiles, a, p, ac, pc) };
    let c = history.entry(key).or_insert(0);
    *c += 1;
    *c
}

/// 각 말이 이번 턴에 둘 수 있는 수가 있는지 (순환 규칙이면 차례가 아닌 말은 false)
fn movable(st: &State, mine: [usize; 2], theirs: [usize; 2], turn: Option<usize>) -> [bool; 2] {
    let mut buf = vec![0i32; bb::MAX_MOVES];
    let (n, _) = bb::generate(st.tiles, State::mask(&mine), State::mask(&theirs), &mut buf);
    let mut out = [false; 2];
    for &m in &buf[..n] {
        for i in 0..2 {
            if bb::src(m) == mine[i] && turn.map_or(true, |t| t == i) {
                out[i] = true;
            }
        }
    }
    out
}

fn read_line(lines: &mut impl Iterator<Item = io::Result<String>>) -> Option<String> {
    io::stdout().flush().ok();
    lines.next().and_then(|l| l.ok()).map(|l| l.trim().to_string())
}

fn player_turn(st: &mut State, lines: &mut impl Iterator<Item = io::Result<String>>) -> Result<Outcome, String> {
    let fixed = st.turn.map(|t| t[0]);
    let can = movable(st, st.player, st.ai, fixed);
    if !can[0] && !can[1] {
        match fixed {
            Some(i) => println!("플레이어 말 {} 은 움직일 수 없어 차례를 넘깁니다.", PLAYER_MARK[i]),
            None => println!("플레이어는 둘 수 있는 수가 없어 차례를 넘깁니다."),
        }
        return Ok(Outcome::Passed);
    }
    loop {
        st.print();
        let idx = if let Some(i) = fixed {
            print!("말 {} 차례 (w/a/s/d): ", PLAYER_MARK[i]);
            i
        } else { loop {
            print!("움직일 말 (1/2): ");
            let Some(l) = read_line(lines) else { return Ok(Outcome::Quit) };
            match l.as_str() {
                "q" => return Ok(Outcome::Quit),
                "1" | "2" => {
                    let i = if l == "1" { 0 } else { 1 };
                    if can[i] {
                        break i;
                    }
                    println!("그 말은 움직일 수 없습니다.");
                }
                _ => println!("1 또는 2를 입력하세요."),
            }
        } };
        let saved = (st.tiles, st.player);
        loop {
            let Some(l) = read_line(lines) else { return Ok(Outcome::Quit) };
            if l == "q" {
                return Ok(Outcome::Quit);
            }
            if l == "r" {
                // 걷다가 막혔거나 마음이 바뀌면 이번 턴을 처음부터 다시 한다
                st.tiles = saved.0;
                st.player = saved.1;
                break;
            }
            let Some(&(_, dir)) = DIRS.iter().find(|(k, _)| *k == l) else {
                println!("Invalid direction (w/a/s/d, r = 이번 턴 처음부터, q = 끝내기)");
                continue;
            };
            let Some(slid) = st.step(true, idx, dir) else { continue };
            if st.player[idx] == GOAL {
                return Ok(Outcome::Won);
            }
            if slid {
                return Ok(Outcome::Moved);
            }
            st.print();
        }
    }
}

fn ai_turn(
    st: &mut State,
    history: &History,
    time_ms: u64,
    threads: usize,
    opts: SearchOptions,
    tune: Tunables,
    weights: &EvalWeights,
    net: &Option<Arc<Network>>,
) -> Outcome {
    let (me, opp) = (State::mask(&st.ai), State::mask(&st.player));
    let (oc, mc) = st.movers();
    let can = movable(st, st.ai, st.player, st.turn.map(|t| t[1]));
    if !can[0] && !can[1] {
        println!("AI는 둘 수 있는 수가 없어 차례를 넘깁니다.");
        return Outcome::Passed;
    }
    let mut ai = SliderAi::with_weights(time_ms, 40, weights);
    ai.set_options(opts);
    ai.set_tunables(tune);
    ai.set_threads(threads);
    ai.set_net(net.clone());
    ai.set_game_history(history.clone());
    let t0 = Instant::now();
    let r = ai.search_turn(st.tiles, me, opp, mc, oc);
    let symbols = bb::to_symbols(st.tiles, me, opp, r.mv);
    let idx = if st.ai[0] == bb::src(r.mv) { 0 } else { 1 };
    let win_rate = if r.score >= WIN_BOUND {
        0.0
    } else if r.score <= -WIN_BOUND {
        1.0
    } else {
        1.0 / (1.0 + (r.score as f64 / 200.0).exp())
    };
    println!(
        "시간차이(m) : {}, 탐색 깊이: {}\nAI 이동: 말 {} {:?}, Player Win rate: {:.1}%",
        t0.elapsed().as_millis(),
        r.depth,
        AI_MARK[idx],
        symbols,
        win_rate * 100.0
    );
    // 고른 수를 사람 조작과 같은 방식(걷기 + 미끄러지기)으로 실행해서 결과를 확인한다
    for s in symbols {
        let dir = DIRS.iter().find(|(k, _)| *k == s).map(|&(_, d)| d).expect("방향");
        let slid = st.step(false, idx, dir).expect("AI 이동을 실행할 수 없습니다");
        if st.ai[idx] == GOAL {
            return Outcome::Won;
        }
        if slid {
            break;
        }
    }
    Outcome::Moved
}

//! AI 끼리 2v2 한 판을 두는 공용 코드. selfplay(데이터 생성)와 matchup(가중치 비교)이 같이 쓴다.
//! 규칙은 실제 게임과 같다: 한 턴에 말 하나, 어느 말이든 골에 멈추면 승리,
//! 둘 수 있는 수가 없으면 차례 넘김, 같은 국면 3번째 반복은 무승부.
//! 규칙은 두 가지 (Rules):
//! - Cycle (기본, 지금 게임): 대각선 배치. 왼쪽 위 쪽1, 오른쪽 위 쪽0, 왼쪽 아래 쪽0, 오른쪽 아래 쪽1.
//!   차례는 쪽0(먼저 두는 쪽, 사람)부터 시계 방향으로 말 하나씩: 오른쪽 위 -> 오른쪽 아래 -> 왼쪽 아래 -> 왼쪽 위 -> ...
//!   그 차례의 말이 움직일 수 없으면 넘긴다 (다음 차례는 순서대로 다음 말).
//! - Free (예전): 쪽 0 이 위쪽 귀퉁이, 쪽 1 이 아래쪽 귀퉁이. 차례마다 자기 말 둘 중 아무거나 움직인다.

use std::sync::Arc;

use crate::ai::{generate_turn, next_turn_pieces, turn_hash, History, SearchOptions, SliderAi, Tunables};
use crate::bitboard::{self as bb, GOAL, MAX_MOVES};
use crate::javarandom::JavaRandom;
use crate::nnue::Network;
use crate::weights::{self, EvalWeights};

pub const START_BOARD: [[u8; 7]; 7] = [
    [1, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1],
];
pub const SIDE0_START: [(usize, usize); 2] = [(0, 0), (0, 6)];
pub const SIDE1_START: [(usize, usize); 2] = [(6, 0), (6, 6)];
/// 순환 규칙의 시작 위치. 각 쪽의 [0] 이 그 쪽에서 먼저 움직이는 말
pub const CYCLE_SIDE0_START: [(usize, usize); 2] = [(0, 6), (6, 0)];
pub const CYCLE_SIDE1_START: [(usize, usize); 2] = [(6, 6), (0, 0)];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rules {
    /// 말 하나씩 시계 방향 순서 (지금 게임)
    Cycle,
    /// 차례마다 자기 말 아무거나 (예전 게임)
    Free,
}

impl Rules {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "cycle" => Ok(Rules::Cycle),
            "free" => Ok(Rules::Free),
            other => Err(format!("규칙은 cycle 또는 free: {}", other)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Rules::Cycle => "cycle",
            Rules::Free => "free",
        }
    }

    /// 시작 국면: (타일, 쪽별 말, 쪽별 다음에 움직일 수 있는 말)
    pub fn start(self) -> (u64, [u64; 2], [u64; 2]) {
        let tiles = bb::from_array(&START_BOARD);
        match self {
            Rules::Free => {
                let p = [bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START)];
                (tiles, p, p)
            }
            Rules::Cycle => {
                let p = [bb::pieces(&CYCLE_SIDE0_START), bb::pieces(&CYCLE_SIDE1_START)];
                let c = [bb::pieces(&CYCLE_SIDE0_START[..1]), bb::pieces(&CYCLE_SIDE1_START[..1])];
                (tiles, p, c)
            }
        }
    }
}

/// 한 쪽의 평가: 선형 가중치 + (선택) 신경망
#[derive(Clone)]
pub struct Player {
    pub weights: EvalWeights,
    pub net: Option<Arc<Network>>,
    pub opts: SearchOptions,
    /// 이 쪽만 다른 수당 시간을 쓸 때 (None 이면 Config 의 시간)
    pub time_ms: Option<u64>,
    /// 한 수를 탐색할 때 쓰는 스레드 수 (병렬 탐색). 기본 1
    pub search_threads: usize,
    pub tune: Tunables,
}

impl Player {
    pub fn new(weights: EvalWeights, net: Option<Arc<Network>>) -> Self {
        Player { weights, net, opts: SearchOptions::default(), time_ms: None, search_threads: 1, tune: Tunables::default() }
    }

    pub fn with_options(mut self, opts: SearchOptions) -> Self {
        self.opts = opts;
        self
    }

    pub fn with_tunables(mut self, tune: Tunables) -> Self {
        self.tune = tune;
        self
    }

    pub fn with_search_threads(mut self, n: usize) -> Self {
        self.search_threads = n.max(1);
        self
    }

    pub fn with_time(mut self, time_ms: Option<u64>) -> Self {
        self.time_ms = time_ms;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub time_ms: u64,
    pub max_depth: i32,
    /// 이 수가 넘도록 안 끝나면 무승부
    pub max_plies: usize,
    /// 처음 몇 수는 무작위로 둬서 시작 국면을 다양하게 한다
    pub random_plies: usize,
    /// 그 뒤에도 이 확률로 무작위 수를 둔다 (데이터 다양성용)
    pub epsilon: f64,
    pub rules: Rules,
}

impl Default for Config {
    fn default() -> Self {
        Config { time_ms: 30, max_depth: 40, max_plies: 150, random_plies: 4, epsilon: 0.0, rules: Rules::Cycle }
    }
}

/// 학습 데이터 한 줄: AI 가 직접 고른 수 앞의 국면.
#[derive(Clone, Debug)]
pub struct Sample {
    pub game: usize,
    pub ply: usize,
    pub side: usize,
    pub features: [i32; weights::COUNT],
    /// 이 국면 (신경망 학습용): 타일, 둘 차례인 쪽 말, 반대편 말
    pub tiles: u64,
    pub me: u64,
    pub opp: u64,
    /// 순환 규칙: 지금 움직일 말 / 상대가 다음에 움직일 말 (자유 규칙이면 me / opp 와 같음)
    pub mc: u64,
    pub oc: u64,
    pub score: i32,
    /// 탐색이 고른 수 (무작위 수로 바꾸기 전). 정책 학습용
    pub mv: i32,
    pub outcome: f64,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    /// -1 무승부, 0 먼저 둔 쪽 승, 1 나중에 둔 쪽 승
    pub winner: i32,
    pub plies: usize,
    pub samples: Vec<Sample>,
}

/// 한 판을 둔다.
///
/// - `players[0]` 이 먼저 두는 쪽의 평가
/// - `opening_seed` 무작위 시작 수의 시드. 같은 값이면 같은 시작 국면이 된다 (Match 에서 쪽을 바꿔 두 번 둘 때 쓴다)
/// - `noise` epsilon 무작위 수에 쓰는 난수 (None 이면 쓰지 않는다)
/// - `record` true 면 학습 데이터를 모은다
pub fn play(
    cfg: &Config,
    players: &[Player; 2],
    opening_seed: i64,
    mut noise: Option<&mut JavaRandom>,
    record: bool,
    game_id: usize,
) -> Outcome {
    let mut opening = JavaRandom::new(opening_seed);
    let (mut tiles, mut pieces, mut cur) = cfg.rules.start();
    let mut counts = History::default();
    *counts.entry(turn_hash(tiles, pieces[0], pieces[1], cur[0], cur[1])).or_insert(0) += 1;
    let mut samples: Vec<Sample> = Vec::new();
    let mut buffer = vec![0i32; MAX_MOVES];

    let mut winner = -1;
    let mut ply = 0usize;
    while ply < cfg.max_plies {
        let side = ply & 1;
        let me = pieces[side];
        let opp = pieces[1 - side];
        let (mc, oc) = (cur[side], cur[1 - side]);
        let (n, _) = generate_turn(tiles, me, opp, mc, &mut buffer);
        if n > 0 {
            let mv;
            if ply < cfg.random_plies {
                mv = buffer[opening.next_int(n as i32) as usize];
            } else {
                let mut engine = SliderAi::with_weights(players[side].time_ms.unwrap_or(cfg.time_ms), cfg.max_depth, &players[side].weights);
                engine.set_net(players[side].net.clone());
                engine.set_options(players[side].opts);
                engine.set_threads(players[side].search_threads);
                engine.set_tunables(players[side].tune);
                engine.set_game_history(counts.clone());
                let result = engine.search_turn(tiles, me, opp, mc, oc);
                let mut chosen = result.mv;
                let immediate_win = bb::dest(chosen) == GOAL;
                if record && !immediate_win {
                    samples.push(Sample {
                        game: game_id,
                        ply,
                        side,
                        features: engine.features_turn(tiles, me, opp, mc, oc),
                        tiles: tiles | me | opp,
                        me,
                        opp,
                        mc,
                        oc,
                        score: result.score,
                        mv: result.mv,
                        outcome: 0.0,
                    });
                }
                if !immediate_win {
                    if let Some(rng) = noise.as_deref_mut() {
                        if rng.next_double() < cfg.epsilon {
                            chosen = buffer[rng.next_int(n as i32) as usize];
                        }
                    }
                }
                mv = chosen;
            }
            tiles = bb::apply(tiles, mv);
            pieces[side] = bb::apply_pieces(me, mv);
            cur[side] = next_turn_pieces(me, mc, pieces[side]);
            if bb::dest(mv) == GOAL {
                winner = side as i32;
                ply += 1;
                break;
            }
        } else {
            cur[side] = next_turn_pieces(me, mc, me);
        }
        let next = 1 - side;
        let seen = {
            let c = counts.entry(turn_hash(tiles, pieces[next], pieces[side], cur[next], cur[side])).or_insert(0);
            *c += 1;
            *c
        };
        if seen >= 3 {
            ply += 1;
            break;
        }
        ply += 1;
    }

    for s in samples.iter_mut() {
        s.outcome = if winner < 0 {
            0.5
        } else if winner as usize == s.side {
            1.0
        } else {
            0.0
        };
    }
    Outcome { winner, plies: ply, samples }
}

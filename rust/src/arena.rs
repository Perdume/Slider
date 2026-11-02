//! AI 끼리 2v2 한 판을 두는 공용 코드. selfplay(데이터 생성)와 matchup(가중치 비교)이 같이 쓴다.
//! 규칙은 실제 게임과 같다: 한 턴에 말 하나, 어느 말이든 골에 멈추면 승리,
//! 둘 수 있는 수가 없으면 차례 넘김, 같은 국면 3번째 반복은 무승부.
//! 쪽(side) 0 이 먼저 두고 위쪽 귀퉁이에서, 쪽 1 이 아래쪽 귀퉁이에서 시작한다.

use std::sync::Arc;

use crate::ai::{position_hash, History, SliderAi};
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

/// 한 쪽의 평가: 선형 가중치 + (선택) 신경망
#[derive(Clone)]
pub struct Player {
    pub weights: EvalWeights,
    pub net: Option<Arc<Network>>,
}

impl Player {
    pub fn new(weights: EvalWeights, net: Option<Arc<Network>>) -> Self {
        Player { weights, net }
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
}

impl Default for Config {
    fn default() -> Self {
        Config { time_ms: 30, max_depth: 40, max_plies: 150, random_plies: 4, epsilon: 0.0 }
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
    pub score: i32,
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
    let mut tiles = bb::from_array(&START_BOARD);
    let mut pieces = [bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START)];
    let mut counts = History::default();
    *counts.entry(position_hash(tiles, pieces[0], pieces[1])).or_insert(0) += 1;
    let mut samples: Vec<Sample> = Vec::new();
    let mut buffer = vec![0i32; MAX_MOVES];

    let mut winner = -1;
    let mut ply = 0usize;
    while ply < cfg.max_plies {
        let side = ply & 1;
        let me = pieces[side];
        let opp = pieces[1 - side];
        let (n, _) = bb::generate(tiles, me, opp, &mut buffer);
        if n > 0 {
            let mv;
            if ply < cfg.random_plies {
                mv = buffer[opening.next_int(n as i32) as usize];
            } else {
                let mut engine = SliderAi::with_weights(cfg.time_ms, cfg.max_depth, &players[side].weights);
                engine.set_net(players[side].net.clone());
                engine.set_game_history(counts.clone());
                let result = engine.search(tiles, me, opp);
                let mut chosen = result.mv;
                let immediate_win = bb::dest(chosen) == GOAL;
                if record && !immediate_win {
                    samples.push(Sample {
                        game: game_id,
                        ply,
                        side,
                        features: engine.features(tiles, me, opp),
                        tiles: tiles | me | opp,
                        me,
                        opp,
                        score: result.score,
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
            if bb::dest(mv) == GOAL {
                winner = side as i32;
                ply += 1;
                break;
            }
        }
        let next = 1 - side;
        let seen = {
            let c = counts.entry(position_hash(tiles, pieces[next], pieces[side])).or_insert(0);
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

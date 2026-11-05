//! 배치 단계 (실험 규칙). 게임 기본 규칙(순환, 말 하나씩 1 -> X -> 2 -> Y)은 그대로 두고
//! 시작 국면만 두 사람이 직접 만든다.
//!
//! - 타일 12개를 쪽0(A) 부터 번갈아 하나씩 놓는다 (각 6개). 가운데 3x3 (골과 그 둘레) 에는 못 놓는다.
//!   가운데 3x3 이 비어 있으면 첫 수에 골에 멈출 수 없다 (골에 멈추려면 골 바로 옆에 막아 줄 타일이 있어야 함).
//! - 그다음 말 4개를 1(A) -> X(B) -> 2(A) -> Y(B) 순서로 말이 없는 타일 위에 놓는다. 대국도 이 순서로 움직인다.
//! - 스왑: 처음에 B 를 맡은 사람이 배치가 끝나기 전 자기(B) 차례에 한 번, 놓는 대신 편을 바꿀 수 있다.
//!   판은 그대로이고, 편을 바꾼 뒤에도 B 가 놓을 차례이므로 이제 B 를 맡은 상대가 이어서 놓는다 (헥스의 스왑과 같음).

use crate::ai::{generate_turn, judge, next_turn_pieces, turn_hash, History, SearchOptions, SliderAi, Tunables};
use crate::bitboard::{self as bb, CELLS, GOAL, MAX_MOVES, SIZE};
use crate::weights::EvalWeights;

pub const TILES: usize = 12;
pub const PIECES: usize = 4;
pub const STEPS: usize = TILES + PIECES;
const TUNED: &str = include_str!("../../weights/tuned.json");

/// 타일을 놓을 수 없는 칸. 환경 변수 SETUP_ZONE 으로 고른다 (실험용):
/// - center (기본): 가운데 3x3 만 금지
/// - edge: 바깥 테두리 24칸에만 놓을 수 있음 (지금 게임처럼 가장자리에서 시작)
pub fn forbidden(cell: usize) -> bool {
    let (r, c) = (cell / SIZE, cell % SIZE);
    match zone() {
        Zone::Edge => !(r == 0 || c == 0 || r == SIZE - 1 || c == SIZE - 1),
        Zone::Center => (2..=4).contains(&r) && (2..=4).contains(&c),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Center,
    Edge,
}

pub fn zone() -> Zone {
    static Z: std::sync::OnceLock<Zone> = std::sync::OnceLock::new();
    *Z.get_or_init(|| match std::env::var("SETUP_ZONE").as_deref() {
        Ok("edge") => Zone::Edge,
        _ => Zone::Center,
    })
}

#[derive(Clone, Debug)]
pub struct Setup {
    pub tiles: u64,
    /// [쪽][말 번호] 칸. 아직 안 놓았으면 usize::MAX
    pub pieces: [[usize; 2]; 2],
    pub step: usize,
    /// owner[쪽] = 그 쪽을 맡은 사람 (0 = 처음에 A 를 맡은 사람, 1 = 처음에 B)
    pub owner: [usize; 2],
    pub swap_used: bool,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { tiles: 0, pieces: [[usize::MAX; 2]; 2], step: 0, owner: [0, 1], swap_used: false }
    }
}

impl Setup {
    pub fn done(&self) -> bool {
        self.step >= STEPS
    }

    /// 지금 놓을 쪽 (0 = A, 1 = B)
    pub fn side(&self) -> usize {
        if self.step < TILES {
            self.step & 1
        } else {
            (self.step - TILES) & 1
        }
    }

    pub fn placing_tile(&self) -> bool {
        self.step < TILES
    }

    pub fn piece_mask(&self) -> u64 {
        let mut m = 0;
        for s in 0..2 {
            for &c in &self.pieces[s] {
                if c != usize::MAX {
                    m |= 1u64 << c;
                }
            }
        }
        m
    }

    /// 지금 놓을 수 있는 칸
    pub fn legal(&self) -> Vec<usize> {
        if self.done() {
            return Vec::new();
        }
        if self.placing_tile() {
            (0..CELLS).filter(|&c| !forbidden(c) && self.tiles >> c & 1 == 0).collect()
        } else {
            let occ = self.piece_mask();
            (0..CELLS).filter(|&c| self.tiles >> c & 1 == 1 && occ >> c & 1 == 0).collect()
        }
    }

    pub fn can_swap(&self) -> bool {
        !self.done() && !self.swap_used && self.side() == 1
    }

    pub fn swap(&mut self) {
        debug_assert!(self.can_swap());
        self.owner.swap(0, 1);
        self.swap_used = true;
    }

    pub fn place(&mut self, cell: usize) {
        if self.placing_tile() {
            self.tiles |= 1u64 << cell;
        } else {
            let k = self.step - TILES;
            self.pieces[k & 1][k >> 1] = cell;
        }
        self.step += 1;
    }

    /// 배치가 끝난 국면: (타일, 쪽별 말, 쪽별 먼저 움직일 말). 대국은 A(쪽0) 의 말 1 부터.
    pub fn start(&self) -> (u64, [u64; 2], [u64; 2]) {
        let mask = |s: usize| (1u64 << self.pieces[s][0]) | (1u64 << self.pieces[s][1]);
        (self.tiles, [mask(0), mask(1)], [1u64 << self.pieces[0][0], 1u64 << self.pieces[1][0]])
    }

    /// 사람이 읽는 판 (# 타일, A/a = 쪽0 말 1/2, B/b = 쪽1 말 X/Y, * 골)
    pub fn diagram(&self) -> String {
        let mut s = String::new();
        for r in 0..SIZE {
            for c in 0..SIZE {
                let cell = r * SIZE + c;
                let ch = if self.pieces[0][0] == cell {
                    '1'
                } else if self.pieces[0][1] == cell {
                    '2'
                } else if self.pieces[1][0] == cell {
                    'X'
                } else if self.pieces[1][1] == cell {
                    'Y'
                } else if self.tiles >> cell & 1 == 1 {
                    '#'
                } else if cell == GOAL {
                    '*'
                } else {
                    '.'
                };
                s.push(ch);
            }
            s.push('\n');
        }
        s
    }
}

/// 게임 설정 엔진 (lmr2,hm + Tunables::game + tuned.json)
pub fn engine(time_ms: u64) -> SliderAi {
    let w = EvalWeights::from_json(TUNED).expect("tuned.json");
    let mut ai = SliderAi::with_weights(time_ms, 40, &w);
    ai.set_options(SearchOptions::parse("lmr2,hm").unwrap_or_default());
    ai.set_tunables(Tunables::game());
    ai
}

/// 배치가 끝난 국면에서 A 쪽 기준 탐색 점수
pub fn eval_start(start: &(u64, [u64; 2], [u64; 2]), time_ms: u64) -> i32 {
    let (tiles, p, c) = *start;
    let mut ai = engine(time_ms);
    ai.search_turn(tiles, p[0], p[1], c[0], c[1]).score
}

#[derive(Clone, Copy, Debug)]
pub struct Played {
    /// -1 무승부, 0 = A 승, 1 = B 승
    pub winner: i32,
    pub plies: usize,
    /// true = 안전장치(max_plies)에 걸린 무승부 (반복 무승부와 구분)
    pub capped: bool,
    /// 둘 수가 없어 넘긴 차례 수
    pub passes: usize,
    /// true = 판정 규칙으로 끝남 (골 도착이 아니라)
    pub judged: bool,
}

/// 끝내는 규칙 실험 (환경 변수): CAP=N 이면 N ply 에 판정 (골에 가까운 쪽 승),
/// SUPERKO=1 이면 이미 나온 국면을 다시 만드는 수 금지 (3회 반복 무승부 대신)
#[derive(Clone, Copy, Debug, Default)]
pub struct Variant {
    pub cap: Option<usize>,
    pub superko: bool,
}

impl Variant {
    pub fn from_env() -> Self {
        Variant {
            cap: std::env::var("CAP").ok().and_then(|v| v.parse().ok()),
            superko: std::env::var("SUPERKO").map(|v| v == "1").unwrap_or(false),
        }
    }
}

/// 주어진 국면에서 두 쪽 모두 같은 엔진으로 끝까지 둔다 (규칙 실험은 Variant::from_env, max_plies 는 안전장치).
pub fn play_from(start: &(u64, [u64; 2], [u64; 2]), time_ms: u64, max_plies: usize) -> Played {
    play_variant(start, time_ms, max_plies, Variant::from_env())
}

pub fn play_variant(start: &(u64, [u64; 2], [u64; 2]), time_ms: u64, max_plies: usize, v: Variant) -> Played {
    let (mut tiles, mut pieces, mut cur) = *start;
    let mut counts = History::default();
    *counts.entry(turn_hash(tiles, pieces[0], pieces[1], cur[0], cur[1])).or_insert(0) += 1;
    let mut buf = vec![0i32; MAX_MOVES];
    let mut ply = 0;
    let mut passes = 0;
    let done = |winner: i32, plies: usize, capped: bool, passes: usize, judged: bool| Played { winner, plies, capped, passes, judged };
    while ply < max_plies {
        if let Some(cap) = v.cap {
            if ply >= cap {
                let j = judge(pieces[0], pieces[1]);
                let w = if j > 0 { 0 } else if j < 0 { 1 } else { -1 };
                return done(w, ply, false, passes, true);
            }
        }
        let side = ply & 1;
        let (me, opp) = (pieces[side], pieces[1 - side]);
        let (mc, oc) = (cur[side], cur[1 - side]);
        let (n, _) = generate_turn(tiles, me, opp, mc, &mut buf);
        let mut moved = false;
        if n > 0 {
            // 실력 차 실험: TIME_A / TIME_B 가 있으면 그 쪽만 수당 시간을 바꾼다
            let t = std::env::var(if side == 0 { "TIME_A" } else { "TIME_B" }).ok().and_then(|x| x.parse().ok()).unwrap_or(time_ms);
            let mut ai = engine(t);
            ai.set_game_history(counts.clone());
            ai.set_cap(v.cap.map(|c| (c - ply) as i32));
            ai.set_superko(v.superko);
            let mv = ai.search_turn(tiles, me, opp, mc, oc).mv;
            let t2 = bb::apply(tiles, mv);
            let p2 = bb::apply_pieces(me, mv);
            let c2 = next_turn_pieces(me, mc, p2);
            let banned = v.superko && counts.contains_key(&turn_hash(t2, opp, p2, oc, c2));
            if mv >= 0 && !banned {
                tiles = t2;
                pieces[side] = p2;
                cur[side] = c2;
                moved = true;
                if bb::dest(mv) == GOAL {
                    return done(side as i32, ply + 1, false, passes, false);
                }
            }
        }
        if !moved {
            passes += 1;
            cur[side] = next_turn_pieces(me, mc, me);
        }
        let next = 1 - side;
        let c = counts.entry(turn_hash(tiles, pieces[next], pieces[side], cur[next], cur[side])).or_insert(0);
        *c += 1;
        // 반복: 보통은 3번째에 무승부, 같은 국면 금지 규칙에서는 (넘김으로) 다시 나오면 진행 불가로 무승부
        if *c >= 3 || (v.superko && *c >= 2) {
            return done(-1, ply + 1, false, passes, false);
        }
        ply += 1;
    }
    done(-1, ply, true, passes, false)
}

/// 간단한 난수 (xorshift64*)
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// 남은 배치를 무작위로 채운다
pub fn random_fill(s: &mut Setup, rng: &mut Rng) {
    while !s.done() {
        let l = s.legal();
        s.place(l[rng.below(l.len())]);
    }
}

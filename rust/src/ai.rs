//! 반복 심화(iterative deepening) + 알파베타 가지치기 negamax 탐색 AI (Java `SliderAI` 의 포팅).
//! 주어진 시간 안에서 가능한 한 깊이 탐색하고, 트랜스포지션 테이블과 킬러/히스토리 휴리스틱으로 수 정렬을 개선한다.
//! 점수는 항상 "지금 둘 차례인 쪽" 기준이다.
//!
//! 한 편은 말을 여러 개(보통 2개) 가진다. 한 턴에 말 하나만 움직이고, 어느 말이든 골에 멈추면 이긴다.
//! 둘 수 있는 수가 없으면 차례를 넘기고, 같은 국면이 3번 나오면 무승부(0점)다.
//!
//! 이 게임은 서로 위협을 막기만 하면 잘 끝나지 않아서, 순수 minimax 는 지지는 않지만 이기려 하지도 않는다.
//! 그래서 안전한 수들 중에서는 "상대가 직접적인 위협에만 반응한다"고 가정한 탐색으로 공격 계획을 세운다.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::{Duration, Instant};

use crate::bitboard::{self as bb, CELLS, GOAL, MAX_MOVES};
use crate::nnue::Network;
use crate::weights::{self, EvalWeights};

/// 키가 이미 `position_hash` 로 섞인 값이라 다시 섞지 않고 그대로 쓰는 해셔 (기본 SipHash 는 노드마다 부르기엔 느리다)
#[derive(Default, Clone, Copy)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) | b as u64;
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

/// 실제 게임에서 나온 국면의 횟수. 키는 `position_hash` 값.
pub type History = HashMap<u64, u32, BuildHasherDefault<IdentityHasher>>;

pub const WIN: i32 = 100_000;
pub const WIN_BOUND: i32 = WIN - 1_000;
/// 상대가 계획을 방해하지 않는다고 가정했을 때 찾은 승리 (확정 승리는 아니다)
pub const PLAN_WIN: i32 = 5_000;
pub const PLAN_BOUND: i32 = PLAN_WIN - 1_000;
pub const DRAW: i32 = 0;
const INF: i32 = WIN + 1;

const MAX_PLY: usize = 64;
/// nnplan 에서 2단계(신경망 평가) 탐색의 최대 깊이. 깊이 4 고정 대국에서 신경망이 선형 평가보다 확실히 강했다
const NN_PLAN_DEPTH: i32 = 4;
const NO_PLAN: i32 = i32::MAX;
const TT_SIZE: usize = 1 << 19;
const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;

/// 최선수와 이 점수 차이 이내면 안전한 수로 보고 공격 계획/함정으로 다시 고른다
const MARGIN: i32 = 40;
/// 상대 응수 중 내가 바로 이기게 되는 비율에 곱하는 보너스
const TRAP_WEIGHT: i32 = 60;
/// 이 ply 부터 상대는 직접적인 위협이 없으면 수를 쉰다고 가정한다 (1, 3 ply 의 상대 응수는 모두 본다)
const PLAN_PLY: i32 = 5;
/// 이기기까지 남은 수(어림)가 이 범위인 직선만 평가에 쓴다 (2수 -> plan2, ... 5수 -> plan5)
const PLAN_MIN_COST: i32 = 2;
const PLAN_MAX_COST: i32 = 5;
const MAX_COST: i32 = 6;

/// 중앙에 가까울수록 높은 위치 점수
const POSITION: [i32; CELLS] = [
    0, 1, 2, 3, 2, 1, 0, //
    1, 3, 4, 5, 4, 3, 1, //
    2, 4, 6, 7, 6, 4, 2, //
    3, 5, 7, 0, 7, 5, 3, //
    2, 4, 6, 7, 6, 4, 2, //
    1, 3, 4, 5, 4, 3, 1, //
    0, 1, 2, 3, 2, 1, 0,
];

/// 골이 있는 줄(가운데 행/열) 위의 칸만 높은 위치 점수. 이 게임은 골과 같은 줄에서 미끄러져야 골에 멈출 수 있어서,
/// 가운데와의 거리보다 "골의 줄 위에 있는가" 가 더 의미 있을 수 있다 (pos_table = 1 로 시험)
const POSITION_LINE: [i32; CELLS] = [
    0, 0, 0, 4, 0, 0, 0, //
    0, 0, 0, 4, 0, 0, 0, //
    0, 0, 0, 4, 0, 0, 0, //
    4, 4, 4, 0, 4, 4, 4, //
    0, 0, 0, 4, 0, 0, 0, //
    0, 0, 0, 4, 0, 0, 0, //
    0, 0, 0, 4, 0, 0, 0,
];
const POSITION_NONE: [i32; CELLS] = [0; CELLS];

const FULL: u64 = (1u64 << CELLS) - 1;
const GOAL_BIT: u64 = 1u64 << GOAL;

struct Tables {
    /// RAY[cell] = 골에서 cell 쪽으로 가는 방향 (골과 같은 행/열이 아니면 -1)
    ray: [i8; CELLS],
    /// BETWEEN[cell] = 골과 cell 사이 칸들
    between: [u64; CELLS],
    /// BACKSTOP[r] = 골에서 r 쪽 직선으로 들어오는 타일을 골에 멈추게 하는 칸
    backstop: [usize; 4],
    col_first: u64,
    col_last: u64,
    border: u64,
}

const fn build_tables() -> Tables {
    let mut ray = [-1i8; CELLS];
    let mut between = [0u64; CELLS];
    let mut backstop = [0usize; 4];
    let (mut col_first, mut col_last, mut border) = (0u64, 0u64, 0u64);
    let size = bb::SIZE;
    let mut cell = 0;
    while cell < CELLS {
        let (r, c) = (cell / size, cell % size);
        if c == 0 {
            col_first |= 1u64 << cell;
        }
        if c == size - 1 {
            col_last |= 1u64 << cell;
        }
        if r == 0 || c == 0 || r == size - 1 || c == size - 1 {
            border |= 1u64 << cell;
        }
        cell += 1;
    }
    let mut r = 0;
    while r < 4 {
        backstop[r] = bb::NEIGHBOR[GOAL][r ^ 1] as usize;
        let mut acc = 0u64;
        let mut cur = bb::NEIGHBOR[GOAL][r] as i32;
        while cur >= 0 {
            ray[cur as usize] = r as i8;
            between[cur as usize] = acc;
            acc |= 1u64 << cur;
            cur = bb::NEIGHBOR[cur as usize][r] as i32;
        }
        r += 1;
    }
    Tables { ray, between, backstop, col_first, col_last, border }
}

static T: Tables = build_tables();

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    /// 고른 이동. 둘 수 있는 수가 없으면 -1 (차례를 넘긴다)
    pub mv: i32,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

/// 탐색 개선 선택지 (체스 엔진에서 쓰는 기법). 모두 꺼져 있으면 Java 엔진과 같은 수를 둔다.
/// 정확 탐색 구간에서만 쓰고, 공격 계획을 찾는 낙관 탐색 구간에는 쓰지 않는다.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchOptions {
    /// PVS: 첫 수 다음부터는 좁은 창으로 "더 나은가"만 확인하고, 그렇다면 다시 탐색한다
    pub pvs: bool,
    /// LMR: 정렬에서 뒤쪽인 조용한 수(TT/킬러가 아닌 4번째 이후)는 한 단계 얕게 먼저 보고, 좋아 보이면 다시 탐색한다
    pub lmr: bool,
    /// LMR 을 더 세게: 깊이 5 이상에서 7번째 이후 수는 2단계 줄인다 (lmr 을 켠 것으로 친다)
    pub lmr2: bool,
    /// 신경망을 후보 고르기에만 쓴다: 1단계(안전한 수 찾기)는 선형 평가로 깊게, 2단계(안전한 수 중 고르기)만
    /// 신경망 평가로 깊이 NN_PLAN_DEPTH 까지. 신경망이 없으면 효과 없음
    pub nnplan: bool,
    /// 널무브: 한 수 쉬어도(차례를 넘겨도) 충분히 좋으면 더 보지 않고 자른다 (상대가 바로 이길 수 있는 국면에서는 안 씀)
    pub nmp: bool,
    /// 위협 정리 탐색 (체스의 quiescence search): 깊이 0 에서 상대의 즉시 승리 위협이 있으면 그걸 막는 수만 더 둬 보고,
    /// 위협이 없는 조용한 국면에서 평가한다. 막을 수 없으면 진 국면
    pub qs: bool,
    /// qs + 위협이 없을 때 "내가 위협을 만드는 수" 도 둬 본다 (체스에서 잡는 수만 더 보는 것과 같음)
    pub qs2: bool,
    /// 응수 수(countermove): 상대의 직전 수에 대해 예전에 잘랐던 수를 킬러 다음으로 먼저 본다
    pub cm: bool,
    /// 연속 히스토리: (상대 직전 수, 내 수) 쌍이 잘랐던 기록을 정렬 점수에 더한다
    pub ch: bool,
    /// 히스토리 감점: 자른 수보다 먼저 봤는데 못 자른 조용한 수는 히스토리 점수를 깎는다
    pub hm: bool,
    /// LMP: 얕은 깊이에서 정렬 뒤쪽의 조용한 수는 아예 보지 않는다 (위협받는 국면 제외)
    pub lmp: bool,
    /// futility: 말단 근처에서 지금 평가 + 여유가 alpha 에도 못 미치면 위협을 만들지 않는 수는 보지 않는다
    pub fut: bool,
    /// 위협 연장: 상대가 바로 이길 수 있는 국면(체스의 체크)은 한 수 더 깊게 본다
    pub ext: bool,
}

impl SearchOptions {
    /// "pvs,lmr,nmp" 같은 쉼표 목록을 읽는다. 빈 문자열이나 "none" 은 모두 끔.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut o = SearchOptions::default();
        for part in text.split(',').map(str::trim).filter(|p| !p.is_empty() && *p != "none") {
            match part {
                "pvs" => o.pvs = true,
                "lmr" => o.lmr = true,
                "lmr2" => {
                    o.lmr = true;
                    o.lmr2 = true;
                }
                "nmp" => o.nmp = true,
                "nnplan" => o.nnplan = true,
                "qs" => o.qs = true,
                "cm" => o.cm = true,
                "ch" => o.ch = true,
                "hm" => o.hm = true,
                "lmp" => o.lmp = true,
                "fut" => o.fut = true,
                "ext" => o.ext = true,
                "qs2" => {
                    o.qs = true;
                    o.qs2 = true;
                }
                other => return Err(format!("알 수 없는 탐색 옵션: {} (pvs, lmr, lmr2, nmp, nnplan, qs, qs2, cm, ch, hm, lmp, fut, ext 중에서)", other)),
            }
        }
        Ok(o)
    }
}

impl std::fmt::Display for SearchOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if self.pvs {
            parts.push("pvs");
        }
        if self.lmr2 {
            parts.push("lmr2");
        } else if self.lmr {
            parts.push("lmr");
        }
        if self.nmp {
            parts.push("nmp");
        }
        if self.qs2 {
            parts.push("qs2");
        } else if self.qs {
            parts.push("qs");
        }
        if self.nnplan {
            parts.push("nnplan");
        }
        for (on, name) in [(self.cm, "cm"), (self.ch, "ch"), (self.hm, "hm"), (self.lmp, "lmp"), (self.fut, "fut"), (self.ext, "ext")] {
            if on {
                parts.push(name);
            }
        }
        write!(f, "{}", if parts.is_empty() { "기본".to_string() } else { parts.join(",") })
    }
}

/// 탐색 상수 (자동 튜닝 대상). 기본값은 예전 고정 상수와 같다.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tunables {
    /// 최선수와 이 점수 차이 이내면 안전한 수
    pub margin: i32,
    /// 함정(상대 응수 중 내가 바로 이기게 되는 비율) 보너스
    pub trap: i32,
    /// 이 ply 부터 상대는 직접적인 위협이 없으면 수를 쉰다고 가정 (공격 계획 탐색)
    pub plan_ply: i32,
    /// LMR: 이 깊이 이상, 이 순서 이상인 조용한 수를 1단계 줄임
    pub lmr_depth: i32,
    pub lmr_index: i32,
    /// lmr2: 이 깊이 이상, 이 순서 이상이면 2단계 줄임
    pub lmr2_depth: i32,
    pub lmr2_index: i32,
    /// 무승부를 수를 고르는 쪽(탐색의 루트)에게만 이만큼 나쁘게 본다 (상대에게는 그만큼 좋게). 0 이면 무승부 = 0점
    pub contempt: i32,
    /// 형세에 따른 contempt: 직전 깊이의 최선 점수가 +dyn_threshold 보다 좋으면(이기는 중) 무승부를 dyn_contempt 만큼 더 나쁘게,
    /// -dyn_threshold 보다 나쁘면(지는 중) 그만큼 좋게 본다. 0 이면 끔
    pub dyn_contempt: i32,
    pub dyn_threshold: i32,
    /// 평가의 위치 점수 표: 0 = 가운데와 가까울수록 (예전), 1 = 골의 줄 위에 있는지, 2 = 없음
    pub pos_table: i32,
    /// LMP: 이 깊이 이하에서 lmp_base + 깊이² 번째 이후 조용한 수는 보지 않는다
    pub lmp_depth: i32,
    pub lmp_base: i32,
    /// futility: 이 깊이 이하에서 평가 + fut_margin × 깊이 <= alpha 이면 자른다
    pub fut_depth: i32,
    pub fut_margin: i32,
    /// 순환 규칙 평가: 내 다른 말(다음다음 차례에 움직일 말)이 지금 골에 멈출 수 있는 수의 개수에 곱하는 점수
    pub nt_me: i32,
    /// 순환 규칙 평가: 상대 다른 말(상대의 그다음 차례 말)이 골에 멈출 수 있는 수의 개수에 곱하는 점수
    pub nt_opp: i32,
    /// 순환 규칙 평가: (내 다른 말 위치 점수 - 상대 다른 말 위치 점수) 에 곱하는 값 (말 순서에 따라 위치를 따로 본다)
    pub pos_nx: i32,
    /// 순환 규칙 평가: 상대가 바로 다음에 움직일 말(oc)의 이동 수에 곱하는 값
    pub mob_oc: i32,
}

impl Default for Tunables {
    fn default() -> Self {
        Tunables { margin: MARGIN, trap: TRAP_WEIGHT, plan_ply: PLAN_PLY, lmr_depth: 3, lmr_index: 3, lmr2_depth: 5, lmr2_index: 6, contempt: 0, dyn_contempt: 0, dyn_threshold: 100, pos_table: 0, lmp_depth: 2, lmp_base: 4, fut_depth: 1, fut_margin: 150, nt_me: 0, nt_opp: 0, pos_nx: 0, mob_oc: 0 }
    }
}

impl Tunables {
    /// 게임용 상수 (새 규칙): VM 에서 상수+가중치를 같이 SPSA 320회로 맞춘 값 (예전 게임 설정 대비 30ms Elo +217, 150ms Elo +211)
    /// + 다음다음 차례 위협 점수 nt_me/nt_opp (단독 +36, 히스토리 감점과 묶어 30ms +31 / 150ms +18). 게임 탐색은 lmr2,hm
    /// + SPSA 3차 (lmr2,hm 에서 상수 9 + 가중치 11, 320회): 그 전 설정 대비 30ms +49, 150ms +40
    pub fn game() -> Self {
        Tunables { margin: 87, trap: 40, plan_ply: 15, lmr_depth: 2, lmr_index: 1, lmr2_depth: 3, lmr2_index: 1, nt_me: 688, nt_opp: -324, ..Tunables::default() }
    }

    /// 예전 규칙(free)용 게임 상수: 예전 규칙에서 SPSA 200회로 맞춘 값 (기본 상수 대비 Elo +50)
    pub fn game_free() -> Self {
        Tunables { margin: 39, trap: 38, plan_ply: 9, lmr_depth: 2, lmr_index: 1, lmr2_depth: 3, lmr2_index: 6, ..Tunables::default() }
    }

    /// "margin=40,trap=60,plan_ply=5,lmr_depth=3,lmr_index=3,lmr2_depth=5,lmr2_index=6" (일부만 줘도 됨)
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut t = Tunables::default();
        for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if part == "game" {
                t = Tunables::game();
                continue;
            }
            if part == "game_free" {
                t = Tunables::game_free();
                continue;
            }
            if part == "default" {
                t = Tunables::default();
                continue;
            }
            let (k, v) = part.split_once('=').ok_or_else(|| format!("이름=값 형식이 아닙니다: {}", part))?;
            let v: i32 = v.trim().parse().map_err(|_| format!("숫자가 아닙니다: {}", part))?;
            match k.trim() {
                "margin" => t.margin = v,
                "trap" => t.trap = v,
                "plan_ply" => t.plan_ply = v.max(1),
                "lmr_depth" => t.lmr_depth = v,
                "lmr_index" => t.lmr_index = v.max(1),
                "lmr2_depth" => t.lmr2_depth = v,
                "lmr2_index" => t.lmr2_index = v.max(1),
                "contempt" => t.contempt = v,
                "dyn_contempt" => t.dyn_contempt = v,
                "dyn_threshold" => t.dyn_threshold = v.max(0),
                "pos_table" => t.pos_table = v.clamp(0, 2),
                "lmp_depth" => t.lmp_depth = v,
                "lmp_base" => t.lmp_base = v.max(1),
                "fut_depth" => t.fut_depth = v,
                "fut_margin" => t.fut_margin = v.max(0),
                "nt_me" => t.nt_me = v,
                "nt_opp" => t.nt_opp = v,
                "pos_nx" => t.pos_nx = v,
                "mob_oc" => t.mob_oc = v,
                other => return Err(format!("알 수 없는 탐색 상수: {}", other)),
            }
        }
        Ok(t)
    }
}

pub struct SliderAi {
    time_limit_ms: u64,
    tune: Tunables,
    /// 지금 쓰는 contempt (= tune.contempt + 형세에 따른 몫)
    cur_contempt: i32,
    opts: SearchOptions,
    max_depth: i32,
    weights: [i32; weights::COUNT],
    /// 있으면 말단 평가에 신경망 출력을 더한다
    net: Option<Arc<Network>>,
    feature: [i32; weights::COUNT],
    /// 순환 규칙 추가 특징 [내 다른 말의 골 수, 상대 다른 말의 골 수] (자유 규칙이면 0)
    cyc: [i32; 4],

    /// 트랜스포지션 테이블 (병렬 탐색 때 스레드끼리 공유). 칸 하나 = [tiles, me, opp, score | move << 32, depth | flag << 8 | mode << 16]
    tt: Arc<SharedTt>,
    /// 한 수를 탐색할 때 쓰는 스레드 수 (Lazy SMP). 1 이면 예전과 똑같이 한 스레드로 탐색한다
    threads: usize,
    /// 0 = 주 스레드, 1.. = 도우미 스레드 (도우미는 깊이를 건너뛰며 탐색해 주 스레드와 다른 곳을 먼저 채운다)
    helper_id: usize,
    /// 주 스레드가 끝나면 도우미를 멈추는 신호
    stop: Option<Arc<AtomicBool>>,

    moves: Vec<Vec<i32>>,
    keys: Vec<Vec<i32>>,
    killers: [[i32; 2]; MAX_PLY],
    history: [i32; 256],
    /// 이 ply 에서 둔 수 (-1 = 차례 넘김). 응수/연속 히스토리에 쓴다
    cur_move: [i32; MAX_PLY],
    /// 상대 직전 수(& 255) -> 그걸 잘 받아친 수
    counter: [i32; 256],
    /// (상대 직전 수 & 255) * 256 + (내 수 & 255) -> 점수
    cont: Vec<i32>,
    scratch: Vec<i32>,
    back_cost: [i32; 4],
    /// 지금 탐색 경로의 국면 해시 (반복 판정용)
    path: [u64; MAX_PLY + 1],
    /// ply 마다: 둘 차례인 쪽이 지금 움직일 수 있는 말 / 상대가 다음 차례에 움직일 수 있는 말
    /// (자유 규칙이면 각각 그쪽 말 전부, 순환 규칙이면 정해진 말 하나)
    mcs: [u64; MAX_PLY + 2],
    ocs: [u64; MAX_PLY + 2],
    scratch2: Vec<i32>,

    /// 실제 게임에서 이미 나온 국면의 횟수 (position_hash 기준). 3번째 반복은 무승부다.
    game_history: History,

    deadline: Instant,
    nodes: u64,
    aborted: bool,
    may_abort: bool,
    opt_ply: i32,
}

/// 스레드끼리 공유하는 트랜스포지션 테이블. 칸마다 값 5개 + 검사값(5개의 XOR)을 원자적으로 쓴다.
/// 여러 스레드가 동시에 같은 칸을 쓰다 섞인 값은 검사값이 맞지 않아 빈 칸으로 본다 (잠금 없이 안전하게).
pub struct SharedTt {
    cells: Vec<AtomicU64>,
}

impl SharedTt {
    fn new(entries: usize) -> Self {
        let n = entries * 6;
        let layout = std::alloc::Layout::array::<AtomicU64>(n).expect("TT 크기");
        // 0으로 채운 메모리를 운영체제에서 바로 받는다 (쓰지 않은 칸은 실제로 건드리지 않아서 빠르다). 0 은 AtomicU64 의 올바른 값이다.
        let cells = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout) as *mut AtomicU64;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(ptr, n, n)
        };
        SharedTt { cells }
    }

    #[inline(always)]
    fn load(&self, idx: usize) -> [u64; 5] {
        let c = &self.cells[idx * 6..idx * 6 + 6];
        let e = [
            c[0].load(Ordering::Relaxed),
            c[1].load(Ordering::Relaxed),
            c[2].load(Ordering::Relaxed),
            c[3].load(Ordering::Relaxed),
            c[4].load(Ordering::Relaxed),
        ];
        if e[0] ^ e[1] ^ e[2] ^ e[3] ^ e[4] == c[5].load(Ordering::Relaxed) {
            e
        } else {
            [0; 5]
        }
    }

    #[inline(always)]
    fn store(&self, idx: usize, e: [u64; 5]) {
        let c = &self.cells[idx * 6..idx * 6 + 6];
        for (cell, v) in c.iter().zip(e.iter()) {
            cell.store(*v, Ordering::Relaxed);
        }
        c[5].store(e[0] ^ e[1] ^ e[2] ^ e[3] ^ e[4], Ordering::Relaxed);
    }
}

/// 국면 해시. me 는 둘 차례인 쪽의 말, opp 는 반대편 말 비트마스크다.
/// 같은 편 말끼리는 순서가 없고, 누구 차례인지도 구분된다.
pub fn position_hash(tiles: u64, me: u64, opp: u64) -> u64 {
    crate::nnue::hash(tiles, me, opp)
}

/// 차례 정보까지 넣은 국면 해시. mc = 지금 움직일 수 있는 내 말, oc = 상대가 다음 차례에 움직일 수 있는 말.
/// 말을 자유롭게 고르는 규칙(mc == me, oc == opp)이면 position_hash 와 같다.
pub fn turn_hash(tiles: u64, me: u64, opp: u64, mc: u64, oc: u64) -> u64 {
    let h = position_hash(tiles, me, opp);
    if mc == me && oc == opp {
        h
    } else {
        h ^ mix(mc.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ oc.rotate_left(29))
    }
}

/// 내가 수를 둔(또는 넘긴) 뒤, 내 다음 차례에 움직일 수 있는 말.
/// 자유 규칙(mc == me)이면 내 말 전부, 순환 규칙(정해진 말 하나)이면 이번에 움직이지 않은 다른 말.
#[inline(always)]
pub fn next_turn_pieces(me_before: u64, mc: u64, me_after: u64) -> u64 {
    if mc == me_before {
        me_after
    } else {
        me_before & !mc
    }
}

/// mc (내 말 중 지금 움직일 수 있는 말) 의 수만 만든다. 다른 내 말은 지나갈 수 없는 칸으로 친다.
#[inline(always)]
pub fn generate_turn(tiles: u64, me: u64, opp: u64, mc: u64, out: &mut [i32]) -> (usize, u64) {
    bb::generate(tiles, mc, opp | (me & !mc), out)
}

/// TT 칸에 넣는 차례 정보 (자유 규칙이면 0)
#[inline(always)]
fn turn_code(me: u64, opp: u64, mc: u64, oc: u64) -> u64 {
    if mc == me && oc == opp {
        0
    } else {
        (mc.trailing_zeros() as u64 + 1) | (oc.trailing_zeros() as u64 + 1) << 8
    }
}

#[inline(always)]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

fn deadline_after(start: Instant, ms: u64) -> Instant {
    start
        .checked_add(Duration::from_millis(ms))
        .unwrap_or_else(|| start + Duration::from_secs(86_400 * 365))
}

impl SliderAi {
    /// 기본 가중치를 쓰는 AI.
    pub fn new(time_limit_ms: u64, max_depth: i32) -> Self {
        Self::with_weights(time_limit_ms, max_depth, &EvalWeights::defaults())
    }

    pub fn with_weights(time_limit_ms: u64, max_depth: i32, w: &EvalWeights) -> Self {
        Self::build(time_limit_ms, max_depth, w.values, Arc::new(SharedTt::new(TT_SIZE)))
    }

    fn build(time_limit_ms: u64, max_depth: i32, weights: [i32; weights::COUNT], tt: Arc<SharedTt>) -> Self {
        SliderAi {
            time_limit_ms,
            tune: Tunables::default(),
            cur_contempt: 0,
            opts: SearchOptions::default(),
            max_depth: max_depth.min(MAX_PLY as i32 - 2),
            weights,
            net: None,
            feature: [0; weights::COUNT],
            cyc: [0; 4],
            tt,
            threads: 1,
            helper_id: 0,
            stop: None,
            moves: (0..MAX_PLY).map(|_| vec![0; MAX_MOVES]).collect(),
            keys: (0..MAX_PLY).map(|_| vec![0; MAX_MOVES]).collect(),
            killers: [[0; 2]; MAX_PLY],
            history: [0; 256],
            cur_move: [-1; MAX_PLY],
            counter: [-1; 256],
            cont: vec![0; 256 * 256],
            scratch: vec![0; MAX_MOVES],
            back_cost: [0; 4],
            path: [0; MAX_PLY + 1],
            mcs: [0; MAX_PLY + 2],
            ocs: [0; MAX_PLY + 2],
            scratch2: vec![0; MAX_MOVES],
            game_history: History::default(),
            deadline: Instant::now(),
            nodes: 0,
            aborted: false,
            may_abort: false,
            opt_ply: NO_PLAN,
        }
    }

    /// 한 수를 탐색할 때 쓸 스레드 수 (기본 1). 2 이상이면 전치표를 공유하는 병렬 탐색(Lazy SMP)을 한다.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// 같은 설정과 전치표를 쓰는 도우미 탐색기
    fn helper(&self, id: usize, stop: Arc<AtomicBool>) -> SliderAi {
        let mut a = SliderAi::build(self.time_limit_ms, self.max_depth, self.weights, self.tt.clone());
        a.net = self.net.clone();
        a.opts = self.opts;
        a.tune = self.tune;
        a.cur_contempt = self.cur_contempt;
        a.game_history = self.game_history.clone();
        a.opt_ply = self.opt_ply;
        a.path[0] = self.path[0];
        a.mcs[0] = self.mcs[0];
        a.ocs[0] = self.ocs[0];
        a.helper_id = id;
        a.stop = Some(stop);
        a
    }

    /// 무승부 점수 (둘 차례인 쪽 기준). 짝수 ply 는 루트(수를 고르는 쪽) 차례라서 -contempt, 홀수 ply 는 상대 차례라서 +contempt.
    #[inline(always)]
    fn draw_score(&self, ply: usize) -> i32 {
        if self.cur_contempt == 0 {
            DRAW
        } else if ply % 2 == 0 {
            DRAW - self.cur_contempt
        } else {
            DRAW + self.cur_contempt
        }
    }

    /// 탐색 상수 (기본은 예전 고정값).
    pub fn set_tunables(&mut self, tune: Tunables) {
        self.tune = tune;
        self.cur_contempt = tune.contempt;
    }

    /// 탐색 개선 선택지 (기본은 모두 끔).
    pub fn set_options(&mut self, opts: SearchOptions) {
        self.opts = opts;
    }

    /// 말단 평가에 더할 신경망 (None 이면 기존 선형 평가만 쓴다).
    pub fn set_net(&mut self, net: Option<Arc<Network>>) {
        self.net = net;
    }

    /// 실제 게임에서 지금까지 나온 국면과 횟수. 키는 position_hash(tiles, 둘 차례인 쪽 말, 반대편 말).
    pub fn set_game_history(&mut self, history: History) {
        self.game_history = history;
    }

    /// me 가 둘 차례인 국면의 평가 특징 (weights::NAMES 순서). 학습 데이터를 만들 때 쓴다.
    /// me 가 바로 이길 수 있는 국면은 평가 대상이 아니므로 호출하는 쪽에서 걸러야 한다.
    pub fn features(&mut self, tiles: u64, me: u64, opp: u64) -> [i32; weights::COUNT] {
        self.features_turn(tiles, me, opp, me, opp)
    }

    /// 순환 규칙용: mc = 지금 움직일 말, oc = 상대가 다음에 움직일 말
    pub fn features_turn(&mut self, tiles: u64, me: u64, opp: u64, mc: u64, oc: u64) -> [i32; weights::COUNT] {
        let tiles = tiles | me | opp;
        let mut list = vec![0i32; MAX_MOVES];
        let (n, reach) = generate_turn(tiles, me, opp, mc, &mut list);
        self.compute_features(tiles, me, opp, mc, oc, reach, &list, n);
        self.feature
    }

    /// 마지막으로 계산한 특징 기준: 지금 / 다음다음 차례 모두 골 위협이 없는 국면
    #[inline]
    pub fn is_quiet(&self) -> bool {
        self.feature[0] == 0 && self.feature[1] == 0 && self.feature[2] == 0 && self.cyc[0] == 0 && self.cyc[1] == 0
    }

    /// 정적 평가 (신경망 제외, 선형 평가 + 순환 규칙 요소). 신경망 학습에서 기준선으로 쓴다
    pub fn static_eval_turn(&mut self, tiles: u64, me: u64, opp: u64, mc: u64, oc: u64) -> i32 {
        let tiles = tiles | me | opp;
        let mut list = vec![0i32; MAX_MOVES];
        let (n, reach) = generate_turn(tiles, me, opp, mc, &mut list);
        let saved = self.net.take();
        let v = self.evaluate(tiles, me, opp, mc, oc, reach, &list, n);
        self.net = saved;
        v
    }

    /// 자식 국면(ply + 1)의 차례 정보를 채운다. me_after = 수를 둔 뒤 내 말 (넘기면 me 그대로)
    #[inline(always)]
    fn set_child(&mut self, ply: usize, me: u64, me_after: u64) {
        self.mcs[ply + 1] = self.ocs[ply];
        self.ocs[ply + 1] = next_turn_pieces(me, self.mcs[ply], me_after);
    }

    /// me 차례에서 최선의 수를 찾는다. 둘 수 있는 수가 없으면 mv 는 -1 (차례를 넘긴다).
    /// 1단계: 정확한 minimax 로 안전한 수(최선과 MARGIN 이내)를 고른다.
    /// 2단계: 안전한 수들만 "상대는 직접적인 위협에만 반응한다"고 가정한 탐색으로 다시 비교해
    ///        여러 수에 걸친 공격 계획이 있는 수를 고른다.
    pub fn search(&mut self, tiles: u64, me: u64, opp: u64) -> SearchResult {
        self.search_turn(tiles, me, opp, me, opp)
    }

    /// 차례 규칙을 지정한 탐색. mc = 지금 움직일 수 있는 내 말, oc = 상대가 다음 차례에 움직일 수 있는 말.
    /// 자유 규칙이면 mc = me, oc = opp (= search). 순환 규칙이면 각각 정해진 말 하나.
    pub fn search_turn(&mut self, tiles: u64, me: u64, opp: u64, mc: u64, oc: u64) -> SearchResult {
        let tiles = tiles | me | opp;
        self.mcs[0] = mc;
        self.ocs[0] = oc;
        let start = Instant::now();
        self.nodes = 0;

        let mut root = vec![0i32; MAX_MOVES];
        let (n, _) = generate_turn(tiles, me, opp, mc, &mut root);
        if n == 0 {
            return SearchResult { mv: -1, score: 0, depth: 0, nodes: 0 };
        }
        for i in 0..n {
            if bb::dest(root[i]) == GOAL {
                return SearchResult { mv: root[i], score: WIN, depth: 1, nodes: 1 };
            }
        }
        self.path[0] = turn_hash(tiles, me, opp, mc, oc);
        root.truncate(n);
        let mut scores = vec![0i32; n];
        self.opt_ply = NO_PLAN;
        let saved_net = if self.opts.nnplan { self.net.take() } else { None };
        let depth = self.iterate(tiles, me, opp, &mut root, &mut scores, deadline_after(start, self.time_limit_ms / 2));
        if self.opts.nnplan {
            self.net = saved_net;
        }
        let best = scores[0];
        let mut safe = 0;
        while safe < n && scores[safe] > best - self.tune.margin {
            safe += 1;
        }
        let mut mv = root[0];
        if best.abs() < WIN_BOUND && safe > 1 {
            let mut plan_moves = root[..safe].to_vec();
            let mut plan_scores = vec![0i32; safe];
            self.opt_ply = self.tune.plan_ply;
            let saved_depth = self.max_depth;
            if self.opts.nnplan && self.net.is_some() {
                self.max_depth = self.max_depth.min(NN_PLAN_DEPTH);
            }
            self.iterate(tiles, me, opp, &mut plan_moves, &mut plan_scores, deadline_after(start, self.time_limit_ms));
            self.max_depth = saved_depth;
            self.opt_ply = NO_PLAN;
            mv = self.choose(tiles, me, opp, &plan_moves, &plan_scores);
        }
        SearchResult { mv, score: best, depth, nodes: self.nodes }
    }

    /// 반복 심화로 root_moves 를 탐색하고 마지막으로 끝낸 깊이의 점수 순서로 정렬한다. 끝낸 깊이를 돌려준다.
    /// 스레드가 2개 이상이면 도우미들이 같은 전치표를 채우며 함께 탐색하고, 결과는 주 스레드의 것을 쓴다.
    fn iterate(&mut self, tiles: u64, me: u64, opp: u64, root_moves: &mut [i32], scores: &mut [i32], until: Instant) -> i32 {
        if self.threads <= 1 {
            return self.iterate_inner(tiles, me, opp, root_moves, scores, until);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut helpers: Vec<SliderAi> = (1..self.threads).map(|id| self.helper(id, stop.clone())).collect();
        let root: Vec<i32> = root_moves.to_vec();
        std::thread::scope(|scope| {
            for h in helpers.iter_mut() {
                let mut moves = root.clone();
                scope.spawn(move || {
                    let mut sc = vec![0i32; moves.len()];
                    h.iterate_inner(tiles, me, opp, &mut moves, &mut sc, until);
                });
            }
            let depth = self.iterate_inner(tiles, me, opp, root_moves, scores, until);
            stop.store(true, Ordering::Relaxed);
            depth
        })
    }

    fn iterate_inner(&mut self, tiles: u64, me: u64, opp: u64, root_moves: &mut [i32], scores: &mut [i32], until: Instant) -> i32 {
        self.deadline = until;
        self.aborted = false;
        let n = root_moves.len();
        let mut work = root_moves.to_vec();
        let mut work_scores = vec![0i32; n];
        let mut done = 0;
        for depth in 1..=self.max_depth {
            // 도우미는 깊이를 하나씩 건너뛰어 주 스레드보다 앞서 깊은 곳의 전치표를 채운다
            if self.helper_id > 0 && depth > 1 && (depth as usize + self.helper_id) % 2 == 0 {
                continue;
            }
            if self.stop.as_ref().map_or(false, |s| s.load(Ordering::Relaxed)) {
                break;
            }
            for h in self.history.iter_mut() {
                *h >>= 1;
            }
            if self.opts.ch {
                for h in self.cont.iter_mut() {
                    *h >>= 1;
                }
            }
            // 깊이 1은 시간이 끝나도 끝까지 탐색해서 항상 결과가 있게 한다
            self.may_abort = depth > 1;
            let mut best = -INF;
            let mut i = 0;
            while i < n && !self.aborted {
                let m = work[i];
                // 최선수와 MARGIN 이내인 수는 정확한 점수를 얻도록 창을 넓힌다
                let floor = if best == -INF { -INF } else { best - self.tune.margin };
                self.cur_move[0] = m;
                self.set_child(0, me, bb::apply_pieces(me, m));
                work_scores[i] =
                    -self.negamax(bb::apply(tiles, m), opp, bb::apply_pieces(me, m), depth - 1, -INF, -floor, 1);
                best = best.max(work_scores[i]);
                i += 1;
            }
            if self.aborted {
                break;
            }
            sort_by_score(&mut work, &mut work_scores, n);
            if self.tune.dyn_contempt != 0 {
                // 다음 깊이는 이번 깊이의 형세로: 이기는 중이면 무승부를 피하고, 지는 중이면 무승부를 반긴다
                let t = self.tune;
                let lean = if best > t.dyn_threshold {
                    t.dyn_contempt
                } else if best < -t.dyn_threshold {
                    -t.dyn_contempt
                } else {
                    0
                };
                self.cur_contempt = t.contempt + lean;
            }
            root_moves.copy_from_slice(&work);
            scores.copy_from_slice(&work_scores);
            done = depth;
            if best.abs() >= PLAN_BOUND || Instant::now() > self.deadline {
                break;
            }
        }
        done
    }

    /// 최선수와 점수 차이가 MARGIN 미만인 안전한 수들 중에서
    /// 상대가 막지 못하면 바로 이길 수 있는 응수의 비율(함정)이 높은 수를 우선한다.
    fn choose(&mut self, tiles: u64, me: u64, opp: u64, root_moves: &[i32], scores: &[i32]) -> i32 {
        let best_score = scores[0];
        if best_score.abs() >= PLAN_BOUND {
            return root_moves[0];
        }
        let mut replies = vec![0i32; MAX_MOVES];
        let mut chosen = root_moves[0];
        let mut chosen_key = f64::NEG_INFINITY;
        let mut i = 0;
        while i < root_moves.len() && scores[i] > best_score - self.tune.margin {
            let m = root_moves[i];
            let after = bb::apply(tiles, m);
            let my_after = bb::apply_pieces(me, m);
            let (oc, nc) = (self.ocs[0], next_turn_pieces(me, self.mcs[0], my_after));
            let (n, _) = generate_turn(after, opp, my_after, oc, &mut replies);
            let mut traps = 0;
            for j in 0..n {
                let r = replies[j];
                if self.can_win(bb::apply(after, r), my_after, bb::apply_pieces(opp, r), nc) {
                    traps += 1;
                }
            }
            let key = scores[i] as f64 + if n == 0 { 0.0 } else { self.tune.trap as f64 * traps as f64 / n as f64 };
            if key > chosen_key {
                chosen_key = key;
                chosen = m;
            }
            i += 1;
        }
        chosen
    }

    /// 위협 정리 탐색. me 차례이고, me 는 바로 이길 수 없다고 가정한다 (호출하는 쪽에서 확인).
    /// list[..n] 은 me 의 수, reach_me 는 me 가 걸어갈 수 있는 칸 (평가에 그대로 쓴다).
    #[allow(clippy::too_many_arguments)]
    fn qsearch_known(&mut self, tiles: u64, me: u64, opp: u64, mut alpha: i32, beta: i32, ply: usize, qply: i32, reach_me: u64, list: &[i32], n: usize) -> i32 {
        const QS_MAX: i32 = 8;
        self.nodes += 1;
        if self.nodes & 1023 == 0
            && self.may_abort
            && (Instant::now() > self.deadline || self.stop.as_ref().map_or(false, |s| s.load(Ordering::Relaxed)))
        {
            self.aborted = true;
        }
        if self.aborted {
            return 0;
        }
        let (mc, oc) = (self.mcs[ply], self.ocs[ply]);
        let threatened = self.can_win(tiles, opp, me, oc);
        if !threatened {
            // 조용한 국면: 평가로 멈출 수 있다 (stand pat)
            let stand = self.evaluate(tiles, me, opp, mc, oc, reach_me, list, n);
            if stand >= beta || !self.opts.qs2 || qply >= QS_MAX || ply >= MAX_PLY - 3 {
                return stand;
            }
            alpha = alpha.max(stand);
            let mut best = stand;
            // 내가 위협을 만드는 수만 더 둬 본다 (상대에게 바로 이기는 수를 주지 않는 것만)
            for i in 0..n {
                let m = list[i];
                let (nt, np) = (bb::apply(tiles, m), bb::apply_pieces(me, m));
                self.set_child(ply, me, np);
                if !self.can_win(nt, np, opp, self.ocs[ply + 1]) || self.can_win(nt, opp, np, self.mcs[ply + 1]) {
                    continue;
                }
                let v = -self.qsearch_child(nt, opp, np, -beta, -alpha, ply + 1, qply + 1);
                if self.aborted {
                    return 0;
                }
                if v > best {
                    best = v;
                }
                if v > alpha {
                    alpha = v;
                }
                if alpha >= beta {
                    break;
                }
            }
            return best;
        }
        if n == 0 {
            // 둘 수 있는 수가 없어 차례를 넘기면 상대가 이긴다
            return -(WIN - (ply as i32 + 1));
        }
        if qply >= QS_MAX || ply >= MAX_PLY - 3 {
            return self.evaluate(tiles, me, opp, mc, oc, reach_me, list, n);
        }
        // 위협을 받는 중: 막는 수만 둬 본다 (막는 수가 없으면 진다)
        let mut best = -(WIN - (ply as i32 + 1));
        for i in 0..n {
            let m = list[i];
            let (nt, np) = (bb::apply(tiles, m), bb::apply_pieces(me, m));
            self.set_child(ply, me, np);
            if self.can_win(nt, opp, np, self.mcs[ply + 1]) {
                continue;
            }
            let v = -self.qsearch_child(nt, opp, np, -beta, -alpha, ply + 1, qply + 1);
            if self.aborted {
                return 0;
            }
            if v > best {
                best = v;
            }
            if v > alpha {
                alpha = v;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// 위협 정리 탐색의 다음 국면 (me 차례, me 는 바로 이길 수 없다고 이미 확인함).
    fn qsearch_child(&mut self, tiles: u64, me: u64, opp: u64, alpha: i32, beta: i32, ply: usize, qply: i32) -> i32 {
        let mut list = std::mem::take(&mut self.moves[ply]);
        let (n, reach_me) = generate_turn(tiles, me, opp, self.mcs[ply], &mut list);
        let v = if list[..n].iter().any(|&m| bb::dest(m) == GOAL) {
            WIN - ply as i32
        } else {
            self.qsearch_known(tiles, me, opp, alpha, beta, ply, qply, reach_me, &list, n)
        };
        self.moves[ply] = list;
        v
    }

    /// me 의 mc 말이 지금 바로 골에 멈출 수 있는가
    fn can_win(&mut self, tiles: u64, me: u64, opp: u64, mc: u64) -> bool {
        let (n, _) = generate_turn(tiles, me, opp, mc, &mut self.scratch);
        self.scratch[..n].iter().any(|&m| bb::dest(m) == GOAL)
    }

    fn negamax(&mut self, tiles: u64, me: u64, opp: u64, depth: i32, alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.nodes & 1023 == 0
            && self.may_abort
            && (Instant::now() > self.deadline || self.stop.as_ref().map_or(false, |s| s.load(Ordering::Relaxed)))
        {
            self.aborted = true;
        }
        if self.aborted {
            return 0;
        }

        // 반복 국면은 무승부다. 같은 차례의 국면은 2 ply 간격으로만 다시 나올 수 있다.
        // 탐색 경로에서 한 번이라도 반복되면 순환으로 보고, 실제 게임에서 이미 2번 나온 국면이면 3번째라서 무승부다.
        let hash = turn_hash(tiles, me, opp, self.mcs[ply], self.ocs[ply]);
        let tcode = turn_code(me, opp, self.mcs[ply], self.ocs[ply]);
        self.path[ply] = hash;
        let mut i = ply as isize - 2;
        while i >= 0 {
            if self.path[i as usize] == hash {
                return self.draw_score(ply);
            }
            i -= 2;
        }
        if !self.game_history.is_empty() && self.game_history.get(&hash).copied().unwrap_or(0) >= 2 {
            return self.draw_score(ply);
        }

        // 같은 국면이라도 탐색 방식(정확/낙관, 낙관까지 남은 수)이 다르면 값이 다르므로 TT 키에 넣는다
        let optimistic = ply as i32 >= self.opt_ply;
        let mode: i32 = if optimistic {
            1 + (ply & 1) as i32
        } else if self.opt_ply == NO_PLAN {
            0
        } else {
            3 + self.opt_ply - ply as i32
        };
        let idx = (mix(hash.wrapping_add(mode as u64)) as usize) & (TT_SIZE - 1);
        let mut tt_best = -1;
        let entry = self.tt.load(idx);
        if entry[1] == me && entry[2] == opp && entry[0] == tiles && ((entry[4] >> 16) & 0xff) as i32 == mode && (entry[4] >> 24) == tcode {
            tt_best = (entry[3] >> 32) as u32 as i32;
            if (entry[4] & 0xff) as u8 as i8 as i32 >= depth {
                let s = from_tt(entry[3] as u32 as i32, ply as i32);
                let flag = ((entry[4] >> 8) & 0xff) as u8;
                if flag == EXACT || (flag == LOWER && s >= beta) || (flag == UPPER && s <= alpha) {
                    return s;
                }
            }
        }

        // 재귀 호출 동안 self 를 빌려줘야 해서 이 층의 버퍼는 잠시 꺼내 쓰고 돌려놓는다
        let mut list = std::mem::take(&mut self.moves[ply]);
        let mut key = std::mem::take(&mut self.keys[ply]);
        let result = self.expand(tiles, me, opp, depth, alpha, beta, ply, optimistic, mode | (tcode << 8) as i32, idx, tt_best, &mut list, &mut key);
        self.moves[ply] = list;
        self.keys[ply] = key;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn expand(
        &mut self,
        tiles: u64,
        me: u64,
        opp: u64,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        optimistic: bool,
        mode: i32,
        idx: usize,
        tt_best: i32,
        list: &mut [i32],
        key: &mut [i32],
    ) -> i32 {
        let (mc, oc) = (self.mcs[ply], self.ocs[ply]);
        let (n, reach_me) = generate_turn(tiles, me, opp, mc, list);
        for i in 0..n {
            if bb::dest(list[i]) == GOAL {
                return (if optimistic { PLAN_WIN } else { WIN }) - ply as i32;
            }
        }
        let t = self.tune;
        // 상대가 바로 이길 수 있는가 (위협 연장 / LMP / futility 에서 쓸 때만 계산한다)
        let need_threat = !optimistic
            && ply < MAX_PLY - 1
            && (self.opts.ext || (self.opts.lmp && depth >= 1 && depth <= t.lmp_depth) || (self.opts.fut && depth >= 1 && depth <= t.fut_depth));
        let threatened = need_threat && self.can_win(tiles, opp, me, oc);
        let mut depth = depth;
        if self.opts.ext && threatened && ply < MAX_PLY - 8 {
            depth += 1;
        }
        if depth <= 0 || ply >= MAX_PLY - 1 {
            if self.opts.qs && !optimistic && ply < MAX_PLY - 2 {
                return self.qsearch_known(tiles, me, opp, alpha, beta, ply, 0, reach_me, list, n);
            }
            return self.evaluate(tiles, me, opp, mc, oc, reach_me, list, n);
        }
        if optimistic && (ply & 1) == 1 && !self.can_win(tiles, opp, me, oc) {
            // 깊은 곳에서 상대는 바로 이기거나 직접적인 위협을 막을 때만 움직인다고 가정한다.
            // 이렇게 해야 AI 가 여러 수에 걸친 공격 계획을 세운다.
            self.cur_move[ply] = -1;
            self.set_child(ply, me, me);
            return -self.negamax(tiles, opp, me, depth - 1, -beta, -alpha, ply + 1);
        }
        if n == 0 {
            // 둘 수 있는 수가 없으면 차례를 넘긴다
            self.cur_move[ply] = -1;
            self.set_child(ply, me, me);
            return -self.negamax(tiles, opp, me, depth - 1, -beta, -alpha, ply + 1);
        }

        // 널무브: 차례를 그냥 넘겨도 beta 이상이면 실제 수는 더 좋을 것으로 보고 자른다.
        // 상대가 바로 이길 수 있는 국면(넘기면 진다)과 승패 점수 근처에서는 쓰지 않는다.
        if self.opts.nmp && !optimistic && depth >= 3 && beta.abs() < WIN_BOUND && !self.can_win(tiles, opp, me, oc) {
            self.cur_move[ply] = -1;
            self.set_child(ply, me, me);
            let v = -self.negamax(tiles, opp, me, depth - 1 - 2, -beta, -beta + 1, ply + 1);
            if self.aborted {
                return 0;
            }
            if v >= beta {
                return beta;
            }
        }

        let prev = if ply >= 1 { self.cur_move[ply - 1] } else { -1 };
        let counter = if self.opts.cm && prev >= 0 { self.counter[(prev & 255) as usize] } else { -1 };
        let cont_base = if self.opts.ch && prev >= 0 { (prev & 255) as usize * 256 } else { usize::MAX };
        for i in 0..n {
            let m = list[i];
            key[i] = if m == tt_best {
                1 << 30
            } else if m == self.killers[ply][0] {
                1 << 29
            } else if m == self.killers[ply][1] {
                1 << 28
            } else if m == counter {
                1 << 27
            } else {
                let c = if cont_base != usize::MAX { self.cont[cont_base + (m & 255) as usize] } else { 0 };
                self.history[(m & 255) as usize] + POSITION[bb::dest(m)] + c
            };
        }

        // futility: 말단 근처에서 지금 평가에 여유를 더해도 alpha 에 못 미치면 위협을 만들지 않는 수는 건너뛴다
        let futile = self.opts.fut
            && !optimistic
            && !threatened
            && depth <= t.fut_depth
            && alpha.abs() < WIN_BOUND
            && self.evaluate(tiles, me, opp, mc, oc, reach_me, list, n) + t.fut_margin * depth <= alpha;
        let lmp = self.opts.lmp && !optimistic && !threatened && depth <= t.lmp_depth;

        let orig_alpha = alpha;
        let mut best = -INF;
        let mut best_move = -1;
        for i in 0..n {
            pick_next(list, key, i, n);
            let m = list[i];
            let late_quiet = i > 0 && key[i] < (1 << 27) && best > -WIN_BOUND;
            if lmp && late_quiet && i as i32 >= t.lmp_base + depth * depth {
                // 남은 수는 모두 정렬 점수가 더 낮은 조용한 수다
                break;
            }
            let (nt, np) = (bb::apply(tiles, m), bb::apply_pieces(me, m));
            self.set_child(ply, me, np);
            if futile && late_quiet && !self.can_win(nt, np, opp, self.ocs[ply + 1]) {
                continue;
            }
            self.cur_move[ply] = m;
            let s = if optimistic || i == 0 || !(self.opts.pvs || self.opts.lmr) {
                -self.negamax(nt, opp, np, depth - 1, -beta, -alpha, ply + 1)
            } else {
                self.search_later_move(nt, opp, np, depth, alpha, beta, ply, i, key[i])
            };
            if self.aborted {
                return 0;
            }
            if s > best {
                best = s;
                best_move = m;
            }
            if s > alpha {
                alpha = s;
            }
            if alpha >= beta {
                if m != self.killers[ply][0] {
                    self.killers[ply][1] = self.killers[ply][0];
                    self.killers[ply][0] = m;
                }
                let h = &mut self.history[(m & 255) as usize];
                *h = (*h + depth * depth).min(1 << 20);
                if self.opts.cm && prev >= 0 {
                    self.counter[(prev & 255) as usize] = m;
                }
                if cont_base != usize::MAX {
                    let c = &mut self.cont[cont_base + (m & 255) as usize];
                    *c = (*c + depth * depth).min(1 << 20);
                }
                if self.opts.hm {
                    // 먼저 봤지만 자르지 못한 수는 감점
                    for j in 0..i {
                        let q = (list[j] & 255) as usize;
                        self.history[q] = (self.history[q] - depth * depth).max(-(1 << 20));
                        if cont_base != usize::MAX {
                            self.cont[cont_base + q] = (self.cont[cont_base + q] - depth * depth).max(-(1 << 20));
                        }
                    }
                }
                break;
            }
        }

        let flag = if best <= orig_alpha {
            UPPER
        } else if best >= beta {
            LOWER
        } else {
            EXACT
        };
        self.tt.store(
            idx,
            [
                tiles,
                me,
                opp,
                to_tt(best, ply as i32) as u32 as u64 | (best_move as u32 as u64) << 32,
                (depth as i8 as u8 as u64) | (flag as u64) << 8 | (mode as u8 as u64) << 16 | ((mode as u64 >> 8) & 0xffff) << 24,
            ],
        );
        best
    }

    /// 정렬에서 첫 수가 아닌 수 (PVS / LMR). 반환값은 negamax 와 같은 의미 (지금 둔 쪽 기준).
    #[allow(clippy::too_many_arguments)]
    fn search_later_move(&mut self, tiles: u64, me: u64, opp: u64, depth: i32, alpha: i32, beta: i32, ply: usize, index: usize, key: i32) -> i32 {
        // 여기서 me 는 수를 둔 뒤 둘 차례가 된 쪽 (상대), opp 는 방금 둔 쪽
        let quiet = key < (1 << 28); // TT 최선수/킬러가 아님
        let t = self.tune;
        let reduce = if self.opts.lmr && depth >= t.lmr_depth && index as i32 >= t.lmr_index && quiet {
            if self.opts.lmr2 && depth >= t.lmr2_depth && index as i32 >= t.lmr2_index { 2 } else { 1 }
        } else {
            0
        };
        if reduce > 0 {
            let v = -self.negamax(tiles, me, opp, depth - 1 - reduce, -alpha - 1, -alpha, ply + 1);
            if self.aborted || v <= alpha {
                return v;
            }
        }
        if self.opts.pvs {
            let v = -self.negamax(tiles, me, opp, depth - 1, -alpha - 1, -alpha, ply + 1);
            if self.aborted || v <= alpha || v >= beta {
                return v;
            }
        }
        -self.negamax(tiles, me, opp, depth - 1, -beta, -alpha, ply + 1)
    }

    /// 말단 평가. me 가 둘 차례이고, me 가 바로 이길 수 없는 상태라고 가정한다.
    #[allow(clippy::too_many_arguments)]
    fn evaluate(&mut self, tiles: u64, me: u64, opp: u64, mc: u64, oc: u64, reach_me: u64, list_me: &[i32], moves_me: usize) -> i32 {
        self.compute_features(tiles, me, opp, mc, oc, reach_me, list_me, moves_me);
        let mut score = 0;
        for i in 0..weights::COUNT {
            score += self.weights[i] * self.feature[i];
        }
        score += self.tune.nt_me * self.cyc[0] + self.tune.nt_opp * self.cyc[1] + self.tune.pos_nx * self.cyc[2] + self.tune.mob_oc * self.cyc[3];
        if let Some(net) = &self.net {
            // 신경망은 조용한 국면(양쪽 다 골 위협 없음)에서만 더한다. 학습도 조용한 국면만으로 한다
            if self.is_quiet() {
                score += net.eval_turn(tiles, me, opp, mc, oc, &self.feature);
            }
        }
        score
    }

    /// 평가 특징을 feature 배열에 채운다 (정의는 weights.rs 참고). scratch 를 덮어쓴다.
    /// 순환 규칙(mc != me)에서는 위협은 상대가 다음에 움직일 말(oc)로만 세고,
    /// 이동 수 / 걸어갈 수 있는 칸 / 계획은 양쪽 말 전부로 센다 (두 번째 말도 곧 움직이므로).
    #[allow(clippy::too_many_arguments)]
    fn compute_features(&mut self, tiles: u64, me: u64, opp: u64, mc: u64, oc: u64, reach_me: u64, list_me: &[i32], moves_me: usize) {
        // 상대 말 전부의 수를 한 번만 만들고, 위협은 그중 oc 말의 골 수만 센다 (oc 말의 수는 따로 만든 것과 같다)
        let (moves_opp, reach_opp) = bb::generate(tiles, opp, me, &mut self.scratch);
        let threats_opp = self.scratch[..moves_opp].iter().filter(|&&m| bb::dest(m) == GOAL && (oc >> bb::src(m)) & 1 != 0).count() as i32;
        let mut list_full = std::mem::take(&mut self.scratch2);
        let (list_me, moves_me, reach_me) = if mc != me {
            // 지금 움직일 말(mc)의 수는 이미 있으니 다른 말의 수만 만들어 붙인다 (말 전부를 새로 만든 것과 같은 목록)
            let (n2, r2) = generate_turn(tiles, me, opp, me & !mc, &mut list_full);
            list_full[n2..n2 + moves_me].copy_from_slice(&list_me[..moves_me]);
            (&list_full[..], n2 + moves_me, r2 | reach_me)
        } else {
            (list_me, moves_me, reach_me)
        };

        self.feature = [0; weights::COUNT];
        if threats_opp == 1 {
            self.feature[weights::THREAT1] = 1; // 막을 수는 있지만 한 수를 강제로 써야 한다
        } else if threats_opp > 1 {
            self.feature[weights::THREAT2] = 1;
            self.feature[weights::THREAT2_PER] = threats_opp;
        }
        self.feature[weights::MOBILITY] = moves_me as i32 - moves_opp as i32;
        self.feature[weights::REACH] = reach_me.count_ones() as i32 - reach_opp.count_ones() as i32;
        backstop_costs(tiles, &mut self.back_cost);
        count_plans(tiles, reach_me, &list_me[..moves_me], &self.back_cost, 1, &mut self.feature);
        count_plans(tiles, reach_opp, &self.scratch[..moves_opp], &self.back_cost, -1, &mut self.feature);
        let table = match self.tune.pos_table {
            1 => &POSITION_LINE,
            2 => &POSITION_NONE,
            _ => &POSITION,
        };
        self.feature[weights::POSITION] = position_score(me, table) - position_score(opp, table);
        self.feature[weights::TEMPO] = 1;
        self.cyc = [0; 4];
        if mc != me {
            // 다음다음 차례의 위협: 내 다른 말 / 상대 다른 말이 지금 판에서 골에 멈출 수 있는 수
            let (mo, oo) = ((me & !mc).trailing_zeros() as usize, (opp & !oc).trailing_zeros() as usize);
            self.cyc[0] = list_me[..moves_me].iter().filter(|&&m| bb::src(m) == mo && bb::dest(m) == GOAL).count() as i32;
            let oci = oc.trailing_zeros() as usize;
            let (mut opp_goal_oo, mut opp_moves_oc) = (0, 0);
            for &m in &self.scratch[..moves_opp] {
                let s = bb::src(m);
                if s == oo && bb::dest(m) == GOAL {
                    opp_goal_oo += 1;
                }
                if s == oci {
                    opp_moves_oc += 1;
                }
            }
            self.cyc[1] = opp_goal_oo;
            self.cyc[2] = table[mo] - table[oo];
            self.cyc[3] = opp_moves_oc;
        }
        self.scratch2 = list_full;
    }
}

fn position_score(pieces: u64, table: &[i32; CELLS]) -> i32 {
    let mut sum = 0;
    let mut p = pieces;
    while p != 0 {
        sum += table[p.trailing_zeros() as usize];
        p &= p - 1;
    }
    sum
}

/// 받침 타일 비용: 골 바로 옆 칸 (골에서 r 반대 방향) 에 타일을 세우려면 몇 개의 타일을 더 옮겨야 하는지.
/// 타일은 다른 타일이나 벽 앞에서만 멈추므로, 타일/벽에서 빈 칸을 따라 잰 거리로 어림한다.
fn backstop_costs(tiles: u64, out: &mut [i32; 4]) {
    let mut solid = tiles;
    let mut layer = (spread(tiles) | T.border) & !tiles & !GOAL_BIT;
    *out = [MAX_COST; 4];
    let mut cost = 1;
    while cost < MAX_COST && layer != 0 {
        for r in 0..4 {
            if out[r] == MAX_COST && (layer >> T.backstop[r]) & 1 != 0 {
                out[r] = cost;
            }
        }
        solid |= layer;
        layer = spread(layer) & !solid & !GOAL_BIT & FULL;
        cost += 1;
    }
    for r in 0..4 {
        if bb::is_tile(tiles, T.backstop[r]) {
            out[r] = 0;
        }
    }
}

fn spread(m: u64) -> u64 {
    ((m << 7) | (m >> 7) | ((m & !T.col_last) << 1) | ((m & !T.col_first) >> 1)) & FULL
}

/// 각 직선(골에서 바깥 방향 r)으로 이기기까지 남은 수를 어림해서, 그 수(2~5)별 직선 개수를 sign 만큼 더한다.
/// 비용 = 받침 타일 비용 + 직선 위 첫 타일을 밟는 비용(0: 이미 밟을 수 있음, 1: 다음 수에 올라감, 2: 그 외) + 1
/// 말이 여러 개면 reach / list 는 모든 말의 합집합이라 가장 가까운 말이 기준이 된다.
fn count_plans(tiles: u64, reach: u64, list: &[i32], back_cost: &[i32; 4], sign: i32, feature: &mut [i32; weights::COUNT]) {
    let mut line_next = 0u32;
    for &m in list {
        let d = bb::dest(m);
        let r = T.ray[d];
        if r >= 0 && ((tiles & !(1u64 << bb::from(m))) & T.between[d]) == 0 {
            line_next |= 1 << r;
        }
    }
    for r in 0..4 {
        let mut cell = bb::neighbor(GOAL, r);
        while cell >= 0 && !bb::is_tile(tiles, cell as usize) {
            cell = bb::neighbor(cell as usize, r);
        }
        let line_cost = if cell >= 0 && bb::is_tile(reach, cell as usize) {
            0
        } else if (line_next >> r) & 1 != 0 {
            1
        } else {
            2
        };
        let cost = back_cost[r] + line_cost + 1;
        if (PLAN_MIN_COST..=PLAN_MAX_COST).contains(&cost) {
            feature[weights::PLAN_FIRST + (cost - PLAN_MIN_COST) as usize] += sign;
        }
    }
}

fn pick_next(list: &mut [i32], key: &mut [i32], from: usize, n: usize) {
    let mut best = from;
    for j in from + 1..n {
        if key[j] > key[best] {
            best = j;
        }
    }
    if best != from {
        list.swap(from, best);
        key.swap(from, best);
    }
}

/// 삽입 정렬 (안정 정렬이라 동점이면 이전 순서를 유지한다)
fn sort_by_score(list: &mut [i32], scores: &mut [i32], n: usize) {
    for i in 1..n {
        let m = list[i];
        let s = scores[i];
        let mut j = i as isize - 1;
        while j >= 0 && scores[j as usize] < s {
            list[j as usize + 1] = list[j as usize];
            scores[j as usize + 1] = scores[j as usize];
            j -= 1;
        }
        list[(j + 1) as usize] = m;
        scores[(j + 1) as usize] = s;
    }
}

fn to_tt(score: i32, ply: i32) -> i32 {
    if score >= PLAN_BOUND {
        score + ply
    } else if score <= -PLAN_BOUND {
        score - ply
    } else {
        score
    }
}

fn from_tt(score: i32, ply: i32) -> i32 {
    if score >= PLAN_BOUND {
        score - ply
    } else if score <= -PLAN_BOUND {
        score + ply
    } else {
        score
    }
}

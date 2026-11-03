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
    /// 널무브: 한 수 쉬어도(차례를 넘겨도) 충분히 좋으면 더 보지 않고 자른다 (상대가 바로 이길 수 있는 국면에서는 안 씀)
    pub nmp: bool,
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
                other => return Err(format!("알 수 없는 탐색 옵션: {} (pvs, lmr, lmr2, nmp 중에서)", other)),
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
        write!(f, "{}", if parts.is_empty() { "기본".to_string() } else { parts.join(",") })
    }
}

pub struct SliderAi {
    time_limit_ms: u64,
    opts: SearchOptions,
    max_depth: i32,
    weights: [i32; weights::COUNT],
    /// 있으면 말단 평가에 신경망 출력을 더한다
    net: Option<Arc<Network>>,
    feature: [i32; weights::COUNT],

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
    scratch: Vec<i32>,
    back_cost: [i32; 4],
    /// 지금 탐색 경로의 국면 해시 (반복 판정용)
    path: [u64; MAX_PLY + 1],

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
            opts: SearchOptions::default(),
            max_depth: max_depth.min(MAX_PLY as i32 - 2),
            weights,
            net: None,
            feature: [0; weights::COUNT],
            tt,
            threads: 1,
            helper_id: 0,
            stop: None,
            moves: (0..MAX_PLY).map(|_| vec![0; MAX_MOVES]).collect(),
            keys: (0..MAX_PLY).map(|_| vec![0; MAX_MOVES]).collect(),
            killers: [[0; 2]; MAX_PLY],
            history: [0; 256],
            scratch: vec![0; MAX_MOVES],
            back_cost: [0; 4],
            path: [0; MAX_PLY + 1],
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
        a.game_history = self.game_history.clone();
        a.opt_ply = self.opt_ply;
        a.path[0] = self.path[0];
        a.helper_id = id;
        a.stop = Some(stop);
        a
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
        let tiles = tiles | me | opp;
        let mut list = vec![0i32; MAX_MOVES];
        let (n, reach) = bb::generate(tiles, me, opp, &mut list);
        self.compute_features(tiles, me, opp, reach, &list, n);
        self.feature
    }

    /// me 차례에서 최선의 수를 찾는다. 둘 수 있는 수가 없으면 mv 는 -1 (차례를 넘긴다).
    /// 1단계: 정확한 minimax 로 안전한 수(최선과 MARGIN 이내)를 고른다.
    /// 2단계: 안전한 수들만 "상대는 직접적인 위협에만 반응한다"고 가정한 탐색으로 다시 비교해
    ///        여러 수에 걸친 공격 계획이 있는 수를 고른다.
    pub fn search(&mut self, tiles: u64, me: u64, opp: u64) -> SearchResult {
        let tiles = tiles | me | opp;
        let start = Instant::now();
        self.nodes = 0;

        let mut root = vec![0i32; MAX_MOVES];
        let (n, _) = bb::generate(tiles, me, opp, &mut root);
        if n == 0 {
            return SearchResult { mv: -1, score: 0, depth: 0, nodes: 0 };
        }
        for i in 0..n {
            if bb::dest(root[i]) == GOAL {
                return SearchResult { mv: root[i], score: WIN, depth: 1, nodes: 1 };
            }
        }
        self.path[0] = position_hash(tiles, me, opp);
        root.truncate(n);
        let mut scores = vec![0i32; n];
        self.opt_ply = NO_PLAN;
        let depth = self.iterate(tiles, me, opp, &mut root, &mut scores, deadline_after(start, self.time_limit_ms / 2));
        let best = scores[0];
        let mut safe = 0;
        while safe < n && scores[safe] > best - MARGIN {
            safe += 1;
        }
        let mut mv = root[0];
        if best.abs() < WIN_BOUND && safe > 1 {
            let mut plan_moves = root[..safe].to_vec();
            let mut plan_scores = vec![0i32; safe];
            self.opt_ply = PLAN_PLY;
            self.iterate(tiles, me, opp, &mut plan_moves, &mut plan_scores, deadline_after(start, self.time_limit_ms));
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
            // 깊이 1은 시간이 끝나도 끝까지 탐색해서 항상 결과가 있게 한다
            self.may_abort = depth > 1;
            let mut best = -INF;
            let mut i = 0;
            while i < n && !self.aborted {
                let m = work[i];
                // 최선수와 MARGIN 이내인 수는 정확한 점수를 얻도록 창을 넓힌다
                let floor = if best == -INF { -INF } else { best - MARGIN };
                work_scores[i] =
                    -self.negamax(bb::apply(tiles, m), opp, bb::apply_pieces(me, m), depth - 1, -INF, -floor, 1);
                best = best.max(work_scores[i]);
                i += 1;
            }
            if self.aborted {
                break;
            }
            sort_by_score(&mut work, &mut work_scores, n);
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
        while i < root_moves.len() && scores[i] > best_score - MARGIN {
            let m = root_moves[i];
            let after = bb::apply(tiles, m);
            let my_after = bb::apply_pieces(me, m);
            let (n, _) = bb::generate(after, opp, my_after, &mut replies);
            let mut traps = 0;
            for j in 0..n {
                let r = replies[j];
                if self.can_win(bb::apply(after, r), my_after, bb::apply_pieces(opp, r)) {
                    traps += 1;
                }
            }
            let key = scores[i] as f64 + if n == 0 { 0.0 } else { TRAP_WEIGHT as f64 * traps as f64 / n as f64 };
            if key > chosen_key {
                chosen_key = key;
                chosen = m;
            }
            i += 1;
        }
        chosen
    }

    fn can_win(&mut self, tiles: u64, me: u64, opp: u64) -> bool {
        let (n, _) = bb::generate(tiles, me, opp, &mut self.scratch);
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
        let hash = position_hash(tiles, me, opp);
        self.path[ply] = hash;
        let mut i = ply as isize - 2;
        while i >= 0 {
            if self.path[i as usize] == hash {
                return DRAW;
            }
            i -= 2;
        }
        if !self.game_history.is_empty() && self.game_history.get(&hash).copied().unwrap_or(0) >= 2 {
            return DRAW;
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
        if entry[1] == me && entry[2] == opp && entry[0] == tiles && ((entry[4] >> 16) & 0xff) as i32 == mode {
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
        let result = self.expand(tiles, me, opp, depth, alpha, beta, ply, optimistic, mode, idx, tt_best, &mut list, &mut key);
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
        let (n, reach_me) = bb::generate(tiles, me, opp, list);
        for i in 0..n {
            if bb::dest(list[i]) == GOAL {
                return (if optimistic { PLAN_WIN } else { WIN }) - ply as i32;
            }
        }
        if depth <= 0 || ply >= MAX_PLY - 1 {
            return self.evaluate(tiles, me, opp, reach_me, list, n);
        }
        if optimistic && (ply & 1) == 1 && !self.can_win(tiles, opp, me) {
            // 깊은 곳에서 상대는 바로 이기거나 직접적인 위협을 막을 때만 움직인다고 가정한다.
            // 이렇게 해야 AI 가 여러 수에 걸친 공격 계획을 세운다.
            return -self.negamax(tiles, opp, me, depth - 1, -beta, -alpha, ply + 1);
        }
        if n == 0 {
            // 둘 수 있는 수가 없으면 차례를 넘긴다
            return -self.negamax(tiles, opp, me, depth - 1, -beta, -alpha, ply + 1);
        }

        // 널무브: 차례를 그냥 넘겨도 beta 이상이면 실제 수는 더 좋을 것으로 보고 자른다.
        // 상대가 바로 이길 수 있는 국면(넘기면 진다)과 승패 점수 근처에서는 쓰지 않는다.
        if self.opts.nmp && !optimistic && depth >= 3 && beta.abs() < WIN_BOUND && !self.can_win(tiles, opp, me) {
            let v = -self.negamax(tiles, opp, me, depth - 1 - 2, -beta, -beta + 1, ply + 1);
            if self.aborted {
                return 0;
            }
            if v >= beta {
                return beta;
            }
        }

        for i in 0..n {
            let m = list[i];
            key[i] = if m == tt_best {
                1 << 30
            } else if m == self.killers[ply][0] {
                1 << 29
            } else if m == self.killers[ply][1] {
                1 << 28
            } else {
                self.history[(m & 255) as usize] + POSITION[bb::dest(m)]
            };
        }

        let orig_alpha = alpha;
        let mut best = -INF;
        let mut best_move = -1;
        for i in 0..n {
            pick_next(list, key, i, n);
            let m = list[i];
            let (nt, np) = (bb::apply(tiles, m), bb::apply_pieces(me, m));
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
                (depth as i8 as u8 as u64) | (flag as u64) << 8 | (mode as u8 as u64) << 16,
            ],
        );
        best
    }

    /// 정렬에서 첫 수가 아닌 수 (PVS / LMR). 반환값은 negamax 와 같은 의미 (지금 둔 쪽 기준).
    #[allow(clippy::too_many_arguments)]
    fn search_later_move(&mut self, tiles: u64, me: u64, opp: u64, depth: i32, alpha: i32, beta: i32, ply: usize, index: usize, key: i32) -> i32 {
        // 여기서 me 는 수를 둔 뒤 둘 차례가 된 쪽 (상대), opp 는 방금 둔 쪽
        let quiet = key < (1 << 28); // TT 최선수/킬러가 아님
        let reduce = if self.opts.lmr && depth >= 3 && index >= 3 && quiet {
            if self.opts.lmr2 && depth >= 5 && index >= 6 { 2 } else { 1 }
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
    fn evaluate(&mut self, tiles: u64, me: u64, opp: u64, reach_me: u64, list_me: &[i32], moves_me: usize) -> i32 {
        self.compute_features(tiles, me, opp, reach_me, list_me, moves_me);
        let mut score = 0;
        for i in 0..weights::COUNT {
            score += self.weights[i] * self.feature[i];
        }
        if let Some(net) = &self.net {
            score += net.eval(tiles, me, opp, &self.feature);
        }
        score
    }

    /// 평가 특징을 feature 배열에 채운다 (정의는 weights.rs 참고). scratch 를 덮어쓴다.
    fn compute_features(&mut self, tiles: u64, me: u64, opp: u64, reach_me: u64, list_me: &[i32], moves_me: usize) {
        let (moves_opp, reach_opp) = bb::generate(tiles, opp, me, &mut self.scratch);
        let threats_opp = self.scratch[..moves_opp].iter().filter(|&&m| bb::dest(m) == GOAL).count() as i32;

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
        self.feature[weights::POSITION] = position_score(me) - position_score(opp);
        self.feature[weights::TEMPO] = 1;
    }
}

fn position_score(pieces: u64) -> i32 {
    let mut sum = 0;
    let mut p = pieces;
    while p != 0 {
        sum += POSITION[p.trailing_zeros() as usize];
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

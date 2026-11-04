//! 브라우저용 엔진 내보내기. 국면은 u64 비트보드 (JS 에서는 BigInt) 로 주고받는다.
//! 칸 번호 = 행 * 7 + 열, tiles 에는 말이 서 있는 칸도 포함된다.
//! 수(i32): from | dir << 6 | dest << 8 | src << 14  (dir: 0 위, 1 아래, 2 왼쪽, 3 오른쪽)
#![allow(static_mut_refs)]

use slider::ai::{generate_turn, turn_hash, History, SearchOptions, SliderAi, Tunables};
use slider::bitboard::{self as bb, CELLS};
use slider::weights::EvalWeights;

const TUNED: &str = include_str!("../../../weights/tuned.json");

static mut MOVES: [i32; bb::MAX_MOVES] = [0; bb::MAX_MOVES];
static mut PATH: [i32; CELLS + 1] = [0; CELLS + 1];
static mut HISTORY: Option<History> = None;
static mut LAST: (i32, i32, f64) = (0, 0, 0.0);

fn history() -> &'static mut History {
    unsafe { HISTORY.get_or_insert_with(History::default) }
}

/// 지금 움직일 말(mc)의 둘 수 있는 수를 moves_ptr() 에 채우고 개수를 돌려준다
#[no_mangle]
pub extern "C" fn gen(tiles: u64, me: u64, opp: u64, mc: u64) -> u32 {
    unsafe { generate_turn(tiles, me, opp, mc, &mut MOVES).0 as u32 }
}

#[no_mangle]
pub extern "C" fn moves_ptr() -> *const i32 {
    unsafe { MOVES.as_ptr() }
}

/// 수를 둔 뒤의 타일
#[no_mangle]
pub extern "C" fn apply_tiles(tiles: u64, mv: i32) -> u64 {
    bb::apply(tiles, mv)
}

/// 국면 해시 (반복 무승부 판정용)
#[no_mangle]
pub extern "C" fn hash(tiles: u64, me: u64, opp: u64, mc: u64, oc: u64) -> u64 {
    turn_hash(tiles, me, opp, mc, oc)
}

/// 말이 걸어간 칸들 (출발 칸부터 미끄러지기 시작한 칸까지) 을 path_ptr() 에 채우고 개수를 돌려준다
#[no_mangle]
pub extern "C" fn path(tiles: u64, me: u64, opp: u64, mv: i32) -> u32 {
    let syms = bb::to_symbols(tiles, me, opp, mv);
    let mut cell = bb::src(mv) as i32;
    unsafe {
        PATH[0] = cell;
        let mut n = 1;
        for s in &syms[..syms.len() - 1] {
            cell += match *s {
                "w" => -7,
                "s" => 7,
                "a" => -1,
                _ => 1,
            };
            PATH[n] = cell;
            n += 1;
        }
        n as u32
    }
}

#[no_mangle]
pub extern "C" fn path_ptr() -> *const i32 {
    unsafe { PATH.as_ptr() }
}

#[no_mangle]
pub extern "C" fn hist_clear() {
    history().clear();
}

#[no_mangle]
pub extern "C" fn hist_add(h: u64) {
    *history().entry(h).or_insert(0) += 1;
}

/// 게임 설정(lmr2,hm + Tunables::game + tuned.json)으로 time_ms 동안 생각해서 수를 돌려준다 (둘 수 없으면 -1).
/// 점수 / 깊이 / 노드는 last_score / last_depth / last_nodes 로 읽는다
#[no_mangle]
pub extern "C" fn think(tiles: u64, me: u64, opp: u64, mc: u64, oc: u64, time_ms: u32) -> i32 {
    let w = EvalWeights::from_json(TUNED).unwrap_or_else(|_| EvalWeights::defaults());
    let mut ai = SliderAi::with_weights(time_ms as u64, 40, &w);
    ai.set_options(SearchOptions::parse("lmr2,hm").unwrap_or_default());
    ai.set_tunables(Tunables::game());
    ai.set_game_history(history().clone());
    let r = ai.search_turn(tiles, me, opp, mc, oc);
    unsafe {
        LAST = (r.score, r.depth, r.nodes as f64);
    }
    r.mv
}

#[no_mangle]
pub extern "C" fn last_score() -> i32 {
    unsafe { LAST.0 }
}

#[no_mangle]
pub extern "C" fn last_depth() -> i32 {
    unsafe { LAST.1 }
}

#[no_mangle]
pub extern "C" fn last_nodes() -> f64 {
    unsafe { LAST.2 }
}

/// 이기는 점수 경계 (이 이상이면 이김 확정)
#[no_mangle]
pub extern "C" fn win_bound() -> i32 {
    slider::ai::WIN_BOUND
}

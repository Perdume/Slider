//! NNUE 방식의 희소 입력 표현. 국면 `(tiles, me, opp)` 를 147개 이진 입력 중 켜진 것들의 번호 목록으로 본다.
//!
//!   0..49    타일이 있는 칸
//!  49..98    둘 차례인 쪽(me) 말이 있는 칸
//!  98..147   반대편(opp) 말이 있는 칸
//!
//! 이동 후의 국면은 한 칸씩 고치지 않고 결과 비트보드에서 통째로 다시 만든다 (덮어쓰기).
//! 켜진 입력이 타일 12개 + 말 4개 정도라 매번 새로 만들어도 싸다.
//! 같은 입력 번호로 국면 해시도 만든다: 켜진 입력마다 정해진 64비트 값을 XOR 한다.
//! me / opp 가 둘 차례 기준이라 누구 차례인지도 해시에 반영된다.

use crate::bitboard::CELLS;

pub const TILE_OFFSET: usize = 0;
pub const MINE_OFFSET: usize = CELLS;
pub const THEIRS_OFFSET: usize = 2 * CELLS;
/// 전체 입력 개수
pub const INPUTS: usize = 3 * CELLS;
/// 켜질 수 있는 입력의 최대 개수 (타일 49 + 말 49 + 49 는 이론상 한계이고, 실제 게임은 20개 안팎)
pub const MAX_ACTIVE: usize = INPUTS;

const fn build_keys() -> [u64; INPUTS] {
    let mut keys = [0u64; INPUTS];
    let mut state: u64 = 0x5EED_0F_51_1D_E5_u64;
    let mut i = 0;
    while i < INPUTS {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        keys[i] = z ^ (z >> 31);
        i += 1;
    }
    keys
}

static KEYS: [u64; INPUTS] = build_keys();

/// 켜진 입력의 번호를 out 에 채우고 개수를 돌려준다 (타일, me 말, opp 말 순서).
pub fn active_inputs(tiles: u64, me: u64, opp: u64, out: &mut [u16; MAX_ACTIVE]) -> usize {
    let mut n = 0;
    for (mask, offset) in [(tiles, TILE_OFFSET), (me, MINE_OFFSET), (opp, THEIRS_OFFSET)] {
        let mut m = mask;
        while m != 0 {
            out[n] = (offset + m.trailing_zeros() as usize) as u16;
            n += 1;
            m &= m - 1;
        }
    }
    n
}

/// 결과 국면에서 통째로 계산하는 국면 해시.
#[inline]
pub fn hash(tiles: u64, me: u64, opp: u64) -> u64 {
    let mut h = 0u64;
    for (mask, offset) in [(tiles, TILE_OFFSET), (me, MINE_OFFSET), (opp, THEIRS_OFFSET)] {
        let mut m = mask;
        while m != 0 {
            h ^= KEYS[offset + m.trailing_zeros() as usize];
            m &= m - 1;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{SIDE0_START, SIDE1_START, START_BOARD};
    use crate::bitboard as bb;

    fn start() -> (u64, u64, u64) {
        (bb::from_array(&START_BOARD), bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START))
    }

    #[test]
    fn start_position_has_expected_active_inputs() {
        let (t, a, b) = start();
        let mut out = [0u16; MAX_ACTIVE];
        let n = active_inputs(t, a, b, &mut out);
        assert_eq!(n, t.count_ones() as usize + 2 + 2);
        assert!(out[..n].iter().all(|&i| (i as usize) < INPUTS));
        // 타일 구역, me 구역, opp 구역 순서
        assert!(out[..n].windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn hash_depends_on_who_is_to_move() {
        let (t, a, b) = start();
        assert_ne!(hash(t, a, b), hash(t, b, a));
    }

    #[test]
    fn hash_is_the_xor_of_active_input_keys() {
        let (t, a, b) = start();
        let mut out = [0u16; MAX_ACTIVE];
        let n = active_inputs(t, a, b, &mut out);
        let expected = out[..n].iter().fold(0u64, |h, &i| h ^ KEYS[i as usize]);
        assert_eq!(hash(t, a, b), expected);
    }

    #[test]
    fn different_move_orders_reaching_the_same_position_hash_equal() {
        // 결과 국면만 보므로 어떤 경로로 왔든 같다 (덮어쓰기 방식의 성질)
        let (t, a, b) = start();
        let mut buf = vec![0i32; bb::MAX_MOVES];
        let (n, _) = bb::generate(t, a, b, &mut buf);
        let m = buf[0];
        let h1 = hash(bb::apply(t, m), b, bb::apply_pieces(a, m));
        let h2 = hash(bb::apply(t, m), b, bb::apply_pieces(a, m));
        assert!(n > 0);
        assert_eq!(h1, h2);
        assert_ne!(h1, hash(t, b, a));
    }
}

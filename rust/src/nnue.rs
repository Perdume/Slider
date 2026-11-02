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


// ---------------------------------------------------------------------------------------------
// 신경망: 147개 희소 입력 -> 은닉 HIDDEN개 (clipped ReLU) -> 출력 1개
// 평가 점수 = (기존 선형 평가) + 신경망 출력 * OUT_SCALE. 출력층을 0으로 시작하므로 학습 전에는 기존 평가와 같다.
// ---------------------------------------------------------------------------------------------

use std::path::Path;

/// 은닉 뉴런 수. 기본 32. 빌드할 때 환경 변수로 바꾼다: `SLIDER_HIDDEN=256 cargo build --release`
/// (신경망 파일 머리말에 크기가 적혀 있어서, 크기가 다른 파일을 읽으면 오류가 난다.)
pub const HIDDEN: usize = hidden_from_env();

const fn hidden_from_env() -> usize {
    match option_env!("SLIDER_HIDDEN") {
        None => 32,
        Some(s) => {
            let b = s.as_bytes();
            let (mut i, mut n) = (0, 0usize);
            while i < b.len() {
                assert!(b[i] >= b'0' && b[i] <= b'9', "SLIDER_HIDDEN 은 숫자여야 한다");
                n = n * 10 + (b[i] - b'0') as usize;
                i += 1;
            }
            assert!(n >= 24 && n <= 4096, "SLIDER_HIDDEN 은 24..4096");
            n
        }
    }
}
/// 신경망 출력(-1..1 근처)을 평가 점수 단위로 바꾸는 배율
pub const OUT_SCALE: f32 = 200.0;

const W1: usize = 0;
const B1: usize = W1 + INPUTS * HIDDEN;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + HIDDEN;
/// 전체 파라미터 개수
pub const PARAMS: usize = B2 + 1;

/// 파라미터를 한 배열에 모아 둔다: [w1 (입력 x 은닉) | b1 | w2 | b2]
#[derive(Clone, Debug)]
pub struct Network {
    pub params: Vec<f32>,
}

impl Network {
    /// 첫 층은 작은 난수, 출력층은 0 (학습 전에는 출력이 항상 0).
    pub fn new(seed: u64) -> Self {
        let mut params = vec![0.0f32; PARAMS];
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for p in params[W1..B1].iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *p = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2; // -0.1 .. 0.1
        }
        for p in params[B1..W2].iter_mut() {
            *p = 0.1;
        }
        Network { params }
    }

    pub fn w1(&self, input: usize) -> &[f32] {
        &self.params[W1 + input * HIDDEN..W1 + (input + 1) * HIDDEN]
    }

    pub fn b1(&self) -> &[f32] {
        &self.params[B1..W2]
    }

    pub fn w2(&self) -> &[f32] {
        &self.params[W2..B2]
    }

    pub fn b2(&self) -> f32 {
        self.params[B2]
    }

    /// 켜진 입력으로 은닉층의 입력값(합)을 새로 계산한다 (증분 갱신이 아니라 통째로 덮어쓴다).
    pub fn accumulate(&self, active: &[u16], acc: &mut [f32; HIDDEN]) {
        acc.copy_from_slice(self.b1());
        for &i in active {
            let w = self.w1(i as usize);
            for j in 0..HIDDEN {
                acc[j] += w[j];
            }
        }
    }

    /// 은닉 합 -> 출력 (clipped ReLU 를 거친 값에 w2 를 곱해 더한다).
    pub fn output(&self, acc: &[f32; HIDDEN]) -> f32 {
        let w2 = self.w2();
        let mut out = self.b2();
        for j in 0..HIDDEN {
            out += w2[j] * acc[j].clamp(0.0, 1.0);
        }
        out
    }

    /// 국면의 신경망 출력 (평가 점수 단위로 환산한 정수). 둘 차례인 쪽(me) 기준.
    pub fn eval(&self, tiles: u64, me: u64, opp: u64) -> i32 {
        let mut active = [0u16; MAX_ACTIVE];
        let n = active_inputs(tiles, me, opp, &mut active);
        let mut acc = [0.0f32; HIDDEN];
        self.accumulate(&active[..n], &mut acc);
        (self.output(&acc) * OUT_SCALE).round() as i32
    }

    /// 텍스트로 저장한다: 머리말 한 줄 + 파라미터 전체 (공백 구분, 값을 그대로 되살릴 수 있는 정밀도).
    pub fn to_text(&self) -> String {
        let mut s = format!("SLIDERNET 1 {} {}\n", INPUTS, HIDDEN);
        let values: Vec<String> = self.params.iter().map(|v| format!("{:?}", v)).collect();
        s.push_str(&values.join(" "));
        s.push('\n');
        s
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let header = lines.next().ok_or("빈 파일입니다")?;
        let expected = format!("SLIDERNET 1 {} {}", INPUTS, HIDDEN);
        if header.trim() != expected {
            return Err(format!("신경망 파일 형식이 맞지 않습니다 (필요: '{}', 실제: '{}')", expected, header.trim()));
        }
        let params: Vec<f32> = lines
            .flat_map(|l| l.split_whitespace())
            .map(|t| t.parse::<f32>().map_err(|_| format!("숫자를 읽을 수 없습니다: {}", t)))
            .collect::<Result<_, _>>()?;
        if params.len() != PARAMS {
            return Err(format!("파라미터 개수가 맞지 않습니다: {} (필요 {})", params.len(), PARAMS));
        }
        Ok(Network { params })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        Self::from_text(&text)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_text()).map_err(|e| format!("{}: {}", path.display(), e))
    }
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

    #[test]
    fn fresh_network_outputs_zero_so_eval_is_unchanged() {
        let (t, a, b) = start();
        assert_eq!(Network::new(1).eval(t, a, b), 0);
    }

    #[test]
    fn network_text_round_trip_is_exact() {
        let mut net = Network::new(3);
        net.params[PARAMS - 1] = 0.123_456_79;
        net.params[B2 - 1] = -1.5e-7;
        let back = Network::from_text(&net.to_text()).unwrap();
        assert_eq!(back.params, net.params);
    }

    #[test]
    fn wrong_header_or_size_is_rejected() {
        assert!(Network::from_text("SLIDERNET 1 3 3\n1 2 3").is_err());
        assert!(Network::from_text(&format!("SLIDERNET 1 {} {}\n1 2 3", INPUTS, HIDDEN)).is_err());
    }

    #[test]
    fn eval_reacts_to_the_board_once_output_weights_are_set() {
        let (t, a, b) = start();
        let mut net = Network::new(5);
        for j in 0..HIDDEN {
            net.params[W2 + j] = 0.5;
        }
        // 출력층이 0이 아니면 은닉층(b1 = 0.1 이상)을 거쳐 출력이 0이 아니게 된다
        assert_ne!(net.eval(t, a, b), 0);
    }
}

//! java.util.Random 과 같은 수열을 내는 난수 생성기 (48비트 LCG).
//! 같은 시드로 Java 와 같은 시작 국면을 만들기 위해 쓴다.

const MULTIPLIER: u64 = 0x5DEE_CE66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1u64 << 48) - 1;

pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    pub fn new(seed: i64) -> Self {
        JavaRandom { seed: (seed as u64 ^ MULTIPLIER) & MASK }
    }

    fn next(&mut self, bits: u32) -> i32 {
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// 0 이상 bound 미만의 정수. bound 는 양수여야 한다.
    pub fn next_int(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive");
        if bound & -bound == bound {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next(31);
            let val = bits % bound;
            if bits.wrapping_sub(val).wrapping_add(bound - 1) >= 0 {
                return val;
            }
        }
    }

    pub fn next_double(&mut self) -> f64 {
        let hi = (self.next(26) as i64) << 27;
        let lo = self.next(27) as i64;
        (hi + lo) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Java 에서 new Random(42) 로 nextInt(100) 을 5번 부른 값
    #[test]
    fn matches_java_sequence_for_seed_42() {
        let mut r = JavaRandom::new(42);
        let got: Vec<i32> = (0..5).map(|_| r.next_int(100)).collect();
        assert_eq!(got, vec![30, 63, 48, 84, 70]);
    }

    #[test]
    fn power_of_two_bound_and_double_stay_in_range() {
        let mut r = JavaRandom::new(7);
        for _ in 0..1000 {
            let v = r.next_int(8);
            assert!((0..8).contains(&v));
            let d = r.next_double();
            assert!((0.0..1.0).contains(&d));
        }
    }
}

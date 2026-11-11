//! 신경망 출력(평가 점수 단위)이 학습 국면과 탐색 중에 나올 법한 국면에서 어느 범위인지 본다.
//!   nn_range <weights.json> <net.txt> <selfplay.jsonl>
use slider::ai::SliderAi;
use slider::arena::{SIDE0_START, SIDE1_START, START_BOARD};
use slider::bitboard as bb;
use slider::nnue::Network;
use slider::weights::{self, EvalWeights};

fn pct(mut v: Vec<i32>) -> String {
    v.sort();
    let p = |q: f64| v[((v.len() - 1) as f64 * q) as usize];
    format!("1% {:>6}  50% {:>6}  99% {:>6}  최소 {:>7}  최대 {:>7}", p(0.01), p(0.5), p(0.99), v[0], v[v.len() - 1])
}

fn num(line: &str, key: &str) -> Option<u64> {
    let i = line.find(&format!("\"{}\":", key))? + key.len() + 3;
    line[i..].split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = EvalWeights::from_argument(Some(&a[1])).unwrap();
    let net = Network::load(std::path::Path::new(&a[2])).unwrap();
    let text = std::fs::read_to_string(&a[3]).unwrap();
    let mut ai = SliderAi::with_weights(10, 1, &w);
    let lin = |f: &[i32; weights::COUNT]| f.iter().zip(w.values.iter()).map(|(x, y)| x * y).sum::<i32>();
    let (mut nn_a, mut lin_a) = (Vec::new(), Vec::new());
    for line in text.lines().take(20000) {
        let (Some(t), Some(me), Some(opp)) = (num(line, "tiles"), num(line, "me"), num(line, "opp")) else { continue };
        let f = ai.features(t, me, opp);
        nn_a.push(net.eval(t, me, opp, &f));
        lin_a.push(lin(&f));
    }
    // 무작위로 둔 국면 (탐색 깊은 곳에서 나올 법한, 실제 게임에서는 드문 국면)
    let (mut nn_b, mut lin_b) = (Vec::new(), Vec::new());
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut buf = vec![0i32; bb::MAX_MOVES];
    while nn_b.len() < 20000 {
        let (mut t, mut me, mut opp) = (bb::from_array(&START_BOARD), bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START));
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let plies = 4 + (rng % 80) as usize;
        let mut ok = true;
        for _ in 0..plies {
            let (n, _) = bb::generate(t, me, opp, &mut buf);
            if n == 0 {
                std::mem::swap(&mut me, &mut opp);
                continue;
            }
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let m = buf[(rng % n as u64) as usize];
            if bb::dest(m) == bb::GOAL {
                ok = false;
                break;
            }
            t = bb::apply(t, m);
            let nm = bb::apply_pieces(me, m);
            me = opp;
            opp = nm;
        }
        if !ok {
            continue;
        }
        let f = ai.features(t, me, opp);
        nn_b.push(net.eval(t, me, opp, &f));
        lin_b.push(lin(&f));
    }
    println!("학습 국면   신경망 {}", pct(nn_a));
    println!("학습 국면   선형   {}", pct(lin_a));
    println!("무작위 국면 신경망 {}", pct(nn_b));
    println!("무작위 국면 선형   {}", pct(lin_b));
}

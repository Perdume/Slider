//! 신경망을 붙였을 때 탐색 속도(초당 노드)와 같은 시간에 닿는 깊이를 비교한다.
//!   eval_speed <weights.json|default> <net.txt|none> [시간ms=30] [국면 수=200]
//! 국면은 시작 국면에서 무작위로 몇 수 둔 것들 (항상 같은 국면).
use slider::ai::SliderAi;
use slider::arena::{SIDE0_START, SIDE1_START, START_BOARD};
use slider::bitboard as bb;
use slider::nnue::Network;
use slider::weights::EvalWeights;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = EvalWeights::from_argument(a.get(1).map(String::as_str)).unwrap();
    let net = match a.get(2).map(String::as_str) {
        None | Some("none") => None,
        Some(p) => Some(Arc::new(Network::load(std::path::Path::new(p)).unwrap())),
    };
    let ms: u64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let count: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut rng = 0x1234_5678_9abc_def1u64;
    let mut buf = vec![0i32; bb::MAX_MOVES];
    let (mut nodes, mut depth, mut secs) = (0u64, 0i64, 0f64);
    let mut done = 0;
    while done < count {
        let (mut t, mut me, mut opp) = (bb::from_array(&START_BOARD), bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START));
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let plies = 4 + (rng % 30) as usize;
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
        let mut ai = SliderAi::with_weights(ms, 40, &w);
        ai.set_net(net.clone());
        let t0 = Instant::now();
        let r = ai.search(t, me, opp);
        secs += t0.elapsed().as_secs_f64();
        nodes += r.nodes as u64;
        depth += r.depth as i64;
        done += 1;
    }
    println!("{}: 국면 {}개, 평균 깊이 {:.2}, 초당 노드 {:.0}", if net.is_some() { "신경망" } else { "선형만" }, done, depth as f64 / done as f64, nodes as f64 / secs);
}

//! 새 규칙(순환)에서 탐색 속도(초당 노드)와 같은 시간에 닿는 깊이를 잰다.
//!   cycle_speed [시간ms=30] [국면 수=100] [탐색 선택지=lmr2] [상수=game]
//! 국면은 시작 국면에서 무작위로 6~20수 둔 것들 (항상 같은 국면).
use slider::ai::{generate_turn, next_turn_pieces, SearchOptions, SliderAi, Tunables};
use slider::arena::Rules;
use slider::bitboard as bb;
use slider::weights::EvalWeights;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ms: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let count: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let opts = SearchOptions::parse(a.get(3).map(String::as_str).unwrap_or("lmr2")).unwrap();
    let tune = Tunables::parse(a.get(4).map(String::as_str).unwrap_or("game")).unwrap();
    let w = EvalWeights::from_json(include_str!("../../weights/tuned.json")).unwrap();
    let mut rng = 0x1234_5678_9abc_def1u64;
    let mut buf = vec![0i32; bb::MAX_MOVES];
    let (mut nodes, mut depth, mut secs, mut done) = (0u64, 0i64, 0f64, 0usize);
    while done < count {
        let (mut t, mut p, mut c) = Rules::Cycle.start();
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let plies = 6 + (rng % 15) as usize;
        let mut ok = true;
        for ply in 0..plies {
            let s = ply & 1;
            let (n, _) = generate_turn(t, p[s], p[1 - s], c[s], &mut buf);
            let before = p[s];
            if n > 0 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = buf[(rng % n as u64) as usize];
                if bb::dest(m) == bb::GOAL {
                    ok = false;
                    break;
                }
                t = bb::apply(t, m);
                p[s] = bb::apply_pieces(p[s], m);
            }
            c[s] = next_turn_pieces(before, c[s], p[s]);
        }
        if !ok {
            continue;
        }
        let s = plies & 1;
        let mut ai = SliderAi::with_weights(ms, 40, &w);
        ai.set_options(opts);
        ai.set_tunables(tune);
        let t0 = Instant::now();
        let r = ai.search_turn(t, p[s], p[1 - s], c[s], c[1 - s]);
        secs += t0.elapsed().as_secs_f64();
        nodes += r.nodes;
        depth += r.depth as i64;
        done += 1;
    }
    println!("국면 {}개, 수당 {}ms: 초당 노드 {:.0}, 평균 깊이 {:.2}", count, ms, nodes as f64 / secs, depth as f64 / count as f64);
}

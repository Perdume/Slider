//! 탐색 선택지(pvs, lmr, nmp)를 기본 탐색과 같은 깊이로 비교한다: 최선 점수가 같은 비율, 노드 수 비율.
//!   search_check <weights.json|default> [깊이=6] [국면 수=150]
use slider::ai::{SearchOptions, SliderAi};
use slider::arena::{SIDE0_START, SIDE1_START, START_BOARD};
use slider::bitboard as bb;
use slider::weights::EvalWeights;

fn positions(count: usize) -> Vec<(u64, u64, u64)> {
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut buf = vec![0i32; bb::MAX_MOVES];
    let mut out = Vec::new();
    while out.len() < count {
        let (mut t, mut me, mut opp) = (bb::from_array(&START_BOARD), bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START));
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let plies = 4 + (rng % 40) as usize;
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
        if ok {
            out.push((t, me, opp));
        }
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = EvalWeights::from_argument(a.get(1).map(String::as_str)).unwrap();
    let depth: i32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let count: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(150);
    let pos = positions(count);
    let run = |opts: SearchOptions| -> Vec<(i32, i32, u64)> {
        pos.iter()
            .map(|&(t, me, opp)| {
                let mut ai = SliderAi::with_weights(1_000_000, depth, &w);
                ai.set_options(opts);
                let r = ai.search(t, me, opp);
                (r.mv, r.score, r.nodes)
            })
            .collect()
    };
    let base = run(SearchOptions::default());
    let base_nodes: u64 = base.iter().map(|r| r.2).sum();
    println!("기본: 깊이 {}, 국면 {}개, 노드 {}", depth, pos.len(), base_nodes);
    for name in ["pvs", "lmr", "nmp", "pvs,lmr", "pvs,lmr,nmp", "qs", "qs2", "lmr2,qs", "lmr2,qs2"] {
        let o = SearchOptions::parse(name).unwrap();
        let r = run(o);
        let same_score = r.iter().zip(&base).filter(|(x, y)| x.1 == y.1).count();
        let same_move = r.iter().zip(&base).filter(|(x, y)| x.0 == y.0).count();
        let nodes: u64 = r.iter().map(|x| x.2).sum();
        println!(
            "{:<12} 최선 점수 같음 {:>3}/{}  수 같음 {:>3}/{}  노드 {:.2}배",
            name, same_score, pos.len(), same_move, pos.len(), nodes as f64 / base_nodes as f64
        );
    }
}

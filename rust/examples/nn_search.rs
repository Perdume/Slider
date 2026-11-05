use std::sync::Arc;
use slider::ai::{SearchOptions, SliderAi, Tunables};
use slider::nnue::Network;
use slider::weights::EvalWeights;
fn num(line: &str, k: &str) -> Option<f64> {
    let p = line.find(&format!("\"{}\":", k))? + k.len() + 3;
    let e = line[p..].find(|c: char| c == ',' || c == '}')? + p;
    line[p..e].parse().ok()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = EvalWeights::from_json(&std::fs::read_to_string(&a[2]).unwrap()).unwrap();
    let net = Arc::new(Network::load(std::path::Path::new(&a[3])).unwrap());
    let text = std::fs::read_to_string(&a[1]).unwrap();
    for line in text.lines().step_by(997).take(8) {
        let (t, me, opp, mc, oc) = (num(line, "tiles").unwrap() as u64, num(line, "me").unwrap() as u64, num(line, "opp").unwrap() as u64, num(line, "mc").unwrap() as u64, num(line, "oc").unwrap() as u64);
        let mut r = vec![];
        for use_net in [false, true] {
            let mut e = SliderAi::with_weights(30, 40, &w);
            e.set_tunables(Tunables::game());
            e.set_options(SearchOptions::parse("lmr2,hm").unwrap());
            if use_net { e.set_net(Some(net.clone())); }
            let s = e.search_turn(t, me, opp, mc, oc);
            let st = e.static_eval_turn(t, me, opp, mc, oc);
            r.push(format!("mv {} sc {} d {} n {} static {}", s.mv, s.score, s.depth, s.nodes, st));
        }
        println!("data {} | {} | {}", num(line, "score").unwrap(), r[0], r[1]);
    }
}

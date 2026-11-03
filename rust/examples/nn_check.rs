//! 신경망 진단: 데이터 국면에서 선형 평가 / 선형+신경망 / 탐색 점수 비교
use std::sync::Arc;
use slider::ai::{SliderAi, Tunables};
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
    let mut e = SliderAi::with_weights(10, 1, &w);
    e.set_tunables(Tunables::game());
    let mut en = SliderAi::with_weights(10, 1, &w);
    en.set_tunables(Tunables::game());
    en.set_net(Some(net.clone()));
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let (mut n, mut s1, mut s2) = (0.0, 0.0, 0.0);
    let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
    for (i, line) in text.lines().enumerate().step_by(7).take(30000) {
        let (t, me, opp, mc, oc, sc) = (num(line, "tiles").unwrap() as u64, num(line, "me").unwrap() as u64, num(line, "opp").unwrap() as u64, num(line, "mc").unwrap() as u64, num(line, "oc").unwrap() as u64, num(line, "score").unwrap());
        if sc.abs() > 4000.0 { continue; }
        let lin = e.static_eval_turn(t, me, opp, mc, oc) as f64;
        let f = e.features_turn(t, me, opp, mc, oc);
        let nn = net.eval_turn(t | me | opp, me, opp, mc, oc, &f) as f64;
        if i < 37 * 15 { println!("score {:6} lin {:6} nn {:6}", sc, lin, nn); }
        xy += lin * sc; xx += lin * lin; yy += sc * sc; n += 1.0; s1 += (lin - sc).abs(); s2 += (lin + nn - sc).abs();
    }
    println!("n {} 평균 |lin-score| {:.0}  |lin+nn-score| {:.0}", n, s1 / n, s2 / n);
    println!("최소제곱 배율 score ~ a*lin: a = {:.3}, 상관 {:.3}", xy / xx, xy / (xx * yy).sqrt());
    let _ = en;
}

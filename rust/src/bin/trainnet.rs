//! 신경망 학습기. `selfplay --positions yes` 로 만든 JSONL 을 읽어 신경망을 학습한다.
//!
//! 예측 확률 p = sigmoid((선형 평가 + 신경망 출력 * OUT_SCALE) / K), 목표 t = λ * sigmoid(탐색점수 / K) + (1 - λ) * 판 결과.
//! 손실은 (p - t)^2 의 평균이고, Adam 으로 역전파한다. 판 단위로 80/20 으로 나눠 검증 손실이 가장 낮았던 시점의 파라미터를 저장한다.
//! K 는 평가 점수 단위의 눈금이라 기본 400 으로 고정한다 (--k 로 바꿀 수 있다).
//! 선형 평가가 판 결과와 거의 무관하면 "판 결과에 맞춘 K" 는 범위 끝으로 가서 sigmoid 가 평평해지고 탐색 점수라는
//! 가장 믿을 만한 신호가 목표에서 사라지므로, 판 결과에 맞춘 K 는 진단용으로만 출력한다.
//!
//! 사용법:
//!   trainnet --data a.jsonl,b.jsonl --weights weights.json --out net.txt
//! 옵션: --data (쉼표로 여러 개) --weights (기본 default) --out --epochs 40 --batch 256 --lr 0.003
//!       --stop-file 경로 / --parent-pid N  멈춤 신호 (멈추면 그때까지의 최선을 저장)
//!       --lambda 0.5 --k 400 --l2 0.0001 --seed 1 --init 이어서 학습할 신경망 파일

use std::collections::HashMap;
use std::path::Path;

use slider::cli;
use slider::nnue::{self, Network, HIDDEN, INPUTS, MAX_ACTIVE, OUT_SCALE, PARAMS};
use slider::weights::{self, EvalWeights};

/// 탐색이 승패를 확정한 국면은 학습에서 뺀다
const SOLVED: f64 = 50_000.0;

struct Row {
    game: usize,
    active: Vec<u16>,
    lin: f32,
    score: f32,
    outcome: f32,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("오류: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = cli::parse_args(&args)?;
    let data = opt.get("data").ok_or("--data 가 필요합니다")?.clone();
    let out = opt.get("out").cloned().unwrap_or_else(|| "net.txt".to_string());
    let weights = EvalWeights::from_argument(opt.get("weights").map(String::as_str))?;
    let epochs: usize = cli::get(&opt, "epochs", 40)?;
    let batch: usize = cli::get(&opt, "batch", 256)?;
    let lr: f32 = cli::get(&opt, "lr", 0.003)?;
    let lambda: f64 = cli::get(&opt, "lambda", 0.5)?;
    let k: f64 = cli::get(&opt, "k", 400.0)?;
    let l2: f32 = cli::get(&opt, "l2", 0.0001)?;
    let seed: u64 = cli::get(&opt, "seed", 1)?;
    let stop = cli::Stop::from_options(&opt)?;

    let mut rows = Vec::new();
    for path in data.split(',') {
        rows.extend(load_rows(path.trim(), &weights)?);
    }
    if rows.len() < 100 {
        return Err(format!("학습 국면이 너무 적습니다: {}", rows.len()));
    }
    // 판 단위로 나눈다 (같은 판의 국면이 학습과 검증에 섞이지 않게)
    let (train, valid): (Vec<&Row>, Vec<&Row>) = rows.iter().partition(|r| r.game % 5 != 0);
    println!("국면 {}개 (학습 {}, 검증 {}), 확정 승패 국면 제외", rows.len(), train.len(), valid.len());

    let k_outcome = calibrate_k(&train);
    println!("눈금 K = {:.0} (고정). 참고: 선형 평가만으로 판 결과를 가장 잘 맞추는 K 는 {:.0}", k, k_outcome);
    if k_outcome > 15_000.0 || k_outcome < 25.0 {
        println!("경고: 판 결과에 맞춘 K 가 탐색 범위 끝입니다. 선형 평가가 판 결과와 거의 무관하다는 뜻이라, 판 결과보다 탐색 점수(--lambda 를 키우기)를 목표로 삼는 편이 낫습니다.");
    }

    let target = |r: &Row| -> f32 { (lambda as f32) * sigmoid(r.score as f64 / k) as f32 + (1.0 - lambda as f32) * r.outcome };

    let mut net = match opt.get("init") {
        Some(path) => Network::load(Path::new(path))?,
        None => Network::new(seed),
    };
    let base_valid = mean_loss(&valid, None, k, &target);
    println!("검증 손실: 선형 평가만 {:.5}  (판 결과 상관 {:.3})", base_valid, corr_with_outcome(&valid));

    let mut best = (mean_loss(&valid, Some(&net), k, &target), net.clone(), 0usize);
    let mut adam = Adam::new();
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for epoch in 1..=epochs {
        if stop.should_stop() {
            println!("중단됨: epoch {} 에서 멈춤", epoch - 1);
            break;
        }
        shuffle(&mut order, &mut rng);
        let mut train_loss = 0.0f64;
        for chunk in order.chunks(batch) {
            let mut grad = vec![0.0f32; PARAMS];
            for &i in chunk {
                let r = train[i];
                train_loss += backprop(&net, r, k, target(r), &mut grad);
            }
            let scale = 1.0 / chunk.len() as f32;
            for g in grad.iter_mut() {
                *g *= scale;
            }
            if l2 > 0.0 {
                let w1_end = INPUTS * HIDDEN;
                let w2_start = w1_end + HIDDEN;
                for i in (0..w1_end).chain(w2_start..w2_start + HIDDEN) {
                    grad[i] += l2 * net.params[i];
                }
            }
            adam.step(&mut net.params, &grad, lr);
        }
        let v = mean_loss(&valid, Some(&net), k, &target);
        if v < best.0 {
            best = (v, net.clone(), epoch);
        }
        if epoch % 5 == 0 || epoch == 1 || epoch == epochs {
            println!("  epoch {:>3}: 학습 {:.5}  검증 {:.5}", epoch, train_loss / train.len() as f64, v);
        }
    }
    println!(
        "가장 좋은 검증 손실 {:.5} (epoch {}), 선형 평가만 쓸 때 {:.5} -> {:+.1}%",
        best.0,
        best.2,
        base_valid,
        (best.0 / base_valid - 1.0) * 100.0
    );
    best.1.save(Path::new(&out))?;
    println!("저장: {}", out);
    Ok(())
}

// ----------------------------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// 한 국면의 손실을 더하고 기울기를 grad 에 쌓는다. 손실값을 돌려준다.
fn backprop(net: &Network, r: &Row, k: f64, t: f32, grad: &mut [f32]) -> f64 {
    let mut acc = [0.0f32; HIDDEN];
    net.accumulate(&r.active, &mut acc);
    let out = net.output(&acc);
    let p = sigmoid(((r.lin + out * OUT_SCALE) as f64) / k) as f32;
    let loss = ((p - t) * (p - t)) as f64;
    // d손실/d출력 = 2 (p - t) p (1 - p) * OUT_SCALE / K
    let d = 2.0 * (p - t) * p * (1.0 - p) * OUT_SCALE / k as f32;

    let w2 = net.w2();
    let (w1_end, b1_end, w2_end) = (INPUTS * HIDDEN, INPUTS * HIDDEN + HIDDEN, INPUTS * HIDDEN + 2 * HIDDEN);
    grad[w2_end] += d; // b2
    for j in 0..HIDDEN {
        let h = acc[j].clamp(0.0, 1.0);
        grad[b1_end + j] += d * h; // w2
        if acc[j] > 0.0 && acc[j] < 1.0 {
            let da = d * w2[j];
            grad[w1_end + j] += da; // b1
            for &i in &r.active {
                grad[i as usize * HIDDEN + j] += da; // w1
            }
        }
    }
    loss
}

fn mean_loss(rows: &[&Row], net: Option<&Network>, k: f64, target: &dyn Fn(&Row) -> f32) -> f64 {
    let mut sum = 0.0;
    for r in rows {
        let out = match net {
            Some(n) => {
                let mut acc = [0.0f32; HIDDEN];
                n.accumulate(&r.active, &mut acc);
                n.output(&acc)
            }
            None => 0.0,
        };
        let p = sigmoid(((r.lin + out * OUT_SCALE) as f64) / k);
        let d = p - target(r) as f64;
        sum += d * d;
    }
    sum / rows.len() as f64
}

/// 선형 평가만으로 판 결과를 가장 잘 맞추는 K (로그 간격 격자 탐색).
fn calibrate_k(rows: &[&Row]) -> f64 {
    let mut best = (f64::INFINITY, 200.0);
    let mut k = 20.0;
    while k <= 20_000.0 {
        let loss: f64 = rows
            .iter()
            .map(|r| {
                let d = sigmoid(r.lin as f64 / k) - r.outcome as f64;
                d * d
            })
            .sum::<f64>()
            / rows.len() as f64;
        if loss < best.0 {
            best = (loss, k);
        }
        k *= 1.12;
    }
    best.1
}

fn corr_with_outcome(rows: &[&Row]) -> f64 {
    let n = rows.len() as f64;
    let (mx, my) = (rows.iter().map(|r| r.lin as f64).sum::<f64>() / n, rows.iter().map(|r| r.outcome as f64).sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for r in rows {
        let (x, y) = (r.lin as f64 - mx, r.outcome as f64 - my);
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
    }
    if sxx == 0.0 || syy == 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    fn new() -> Self {
        Adam { m: vec![0.0; PARAMS], v: vec![0.0; PARAMS], t: 0 }
    }

    fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        self.t += 1;
        let c1 = 1.0 - B1.powi(self.t);
        let c2 = 1.0 - B2.powi(self.t);
        for i in 0..params.len() {
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grad[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grad[i] * grad[i];
            params[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

fn shuffle(order: &mut [usize], state: &mut u64) {
    for i in (1..order.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let j = (*state % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

// ----------------------------------------------------------------------------------------------

fn load_rows(path: &str, weights: &EvalWeights) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f = fields(line);
        let get = |k: &str| -> Result<f64, String> {
            f.get(k).and_then(|v| v.parse::<f64>().ok()).ok_or_else(|| {
                format!("{}:{}: '{}' 를 읽을 수 없습니다 (selfplay 를 --positions yes 로 돌렸는지 확인하세요)", path, n + 1, k)
            })
        };
        let score = get("score")?;
        if score.abs() >= SOLVED {
            skipped += 1;
            continue;
        }
        let features = array_field(line, "features").ok_or_else(|| format!("{}:{}: features 가 없습니다", path, n + 1))?;
        if features.len() != weights::COUNT {
            return Err(format!("{}:{}: features 개수가 {} 가 아닙니다", path, n + 1, weights::COUNT));
        }
        let lin: f64 = features.iter().zip(weights.values.iter()).map(|(f, w)| f * *w as f64).sum();
        let (tiles, me, opp) = (get("tiles")? as u64, get("me")? as u64, get("opp")? as u64);
        let mut buf = [0u16; MAX_ACTIVE];
        let count = nnue::active_inputs(tiles, me, opp, &mut buf);
        rows.push(Row {
            // game 번호는 파일마다 0부터 시작하므로 파일을 구분하려고 큰 수를 더하지 않고 그대로 쓴다 (80/20 분할용)
            game: get("game")? as usize,
            active: buf[..count].to_vec(),
            lin: lin as f32,
            score: score as f32,
            outcome: get("outcome")? as f32,
        });
    }
    println!("{}: 국면 {}개 읽음 (확정 승패 {}개 제외)", path, rows.len(), skipped);
    Ok(rows)
}

/// 한 줄 JSON 에서 `"이름":숫자` 를 모은다 (배열 값은 건너뛴다).
fn fields(line: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            if j + 1 < bytes.len() && bytes[j + 1] == b':' {
                let key = line[start..j].to_string();
                let mut k = j + 2;
                let vstart = k;
                if k < bytes.len() && bytes[k] == b'[' {
                    while k < bytes.len() && bytes[k] != b']' {
                        k += 1;
                    }
                } else {
                    while k < bytes.len() && bytes[k] != b',' && bytes[k] != b'}' {
                        k += 1;
                    }
                    map.insert(key, line[vstart..k].to_string());
                }
                i = k;
                continue;
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    map
}

fn array_field(line: &str, key: &str) -> Option<Vec<f64>> {
    let pat = format!("\"{}\":[", key);
    let start = line.find(&pat)? + pat.len();
    let end = start + line[start..].find(']')?;
    line[start..end].split(',').map(|x| x.trim().parse::<f64>().ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row() -> Row {
        // 시작 국면과 비슷한 입력 몇 개
        Row { game: 0, active: vec![0, 1, 5, 6, 43, 48, 49 + 0, 49 + 6, 98 + 42, 98 + 48], lin: 35.0, score: 20.0, outcome: 1.0 }
    }

    /// 역전파한 기울기가 수치 미분(중앙 차분)과 맞는지 본다.
    #[test]
    fn backprop_matches_numerical_gradient() {
        let mut net = Network::new(9);
        // 출력층을 0이 아니게 해서 첫 층까지 기울기가 흐르게 한다
        let w2_start = INPUTS * HIDDEN + HIDDEN;
        for j in 0..HIDDEN {
            net.params[w2_start + j] = 0.3 - 0.02 * j as f32;
        }
        net.params[PARAMS - 1] = 0.05;
        let r = sample_row();
        let (k, t) = (200.0, 0.7f32);

        let mut grad = vec![0.0f32; PARAMS];
        backprop(&net, &r, k, t, &mut grad);

        let loss_at = |n: &Network| -> f64 {
            let mut g = vec![0.0f32; PARAMS];
            backprop(n, &r, k, t, &mut g)
        };
        let eps = 1e-2f32;
        let mut checked = 0;
        // 활성 입력의 w1, b1, w2, b2 에서 골고루 확인한다
        let mut indices: Vec<usize> = Vec::new();
        for &i in &r.active {
            indices.push(i as usize * HIDDEN + 3);
            indices.push(i as usize * HIDDEN + 17);
        }
        indices.extend([INPUTS * HIDDEN + 3, INPUTS * HIDDEN + 20, w2_start + 3, w2_start + 20, PARAMS - 1]);
        for idx in indices {
            let mut plus = net.clone();
            plus.params[idx] += eps;
            let mut minus = net.clone();
            minus.params[idx] -= eps;
            let numeric = (loss_at(&plus) - loss_at(&minus)) / (2.0 * eps as f64);
            let analytic = grad[idx] as f64;
            let tol = 1e-4 + 0.05 * numeric.abs().max(analytic.abs());
            assert!((numeric - analytic).abs() <= tol, "param {}: numeric {} vs analytic {}", idx, numeric, analytic);
            if analytic.abs() > 1e-7 {
                checked += 1;
            }
        }
        assert!(checked >= 6, "기울기가 0이 아닌 비교 대상이 너무 적습니다: {}", checked);
    }

    #[test]
    fn fields_and_array_parsing() {
        let line = r#"{"game":3,"ply":4,"side":0,"features":[1,-2,3],"score":-45,"outcome":0.5,"tiles":99}"#;
        let f = fields(line);
        assert_eq!(f.get("game").map(String::as_str), Some("3"));
        assert_eq!(f.get("outcome").map(String::as_str), Some("0.5"));
        assert_eq!(f.get("tiles").map(String::as_str), Some("99"));
        assert!(f.get("features").is_none());
        assert_eq!(array_field(line, "features"), Some(vec![1.0, -2.0, 3.0]));
    }
}

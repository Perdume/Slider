//! GPU(JAX) 규칙 구현과 맞춰 보기 위한 도구.
//!   rules_dump perft 5            시작 국면에서 깊이별 노드 수 (통과 = 1수, 골 도달은 거기서 끝)
//!   rules_dump positions 300 7    무작위 대국 300판에서 국면과 가능한 수 목록 (JSON 한 줄씩)
//!   rules_dump playouts 20000 1   무작위 대국 N판의 속도 (CPU 한 스레드 기준값)
//!   rules_dump games 20 20        정해진 규칙(ply*a+b 번째 수)으로 둔 대국 결과 (a 1..=20, b 0..20). 반복 무승부 포함
use slider::arena::{SIDE0_START, SIDE1_START, START_BOARD};
use slider::bitboard as bb;
use std::collections::HashMap;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn start() -> (u64, u64, u64) {
    (bb::from_array(&START_BOARD), bb::pieces(&SIDE0_START), bb::pieces(&SIDE1_START))
}

fn perft(tiles: u64, me: u64, opp: u64, depth: u32, buf: &mut [i32]) -> u64 {
    if depth == 0 {
        return 1;
    }
    let (n, _) = bb::generate(tiles, me, opp, buf);
    if n == 0 {
        return perft(tiles, opp, me, depth - 1, buf);
    }
    let moves: Vec<i32> = buf[..n].to_vec();
    let mut total = 0;
    for m in moves {
        if bb::dest(m) == bb::GOAL {
            total += 1;
        } else {
            total += perft(bb::apply(tiles, m), opp, bb::apply_pieces(me, m), depth - 1, buf);
        }
    }
    total
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("perft");
    let a: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut buf = vec![0i32; bb::MAX_MOVES];
    match mode {
        "perft" => {
            let (t, m, o) = start();
            for d in 1..=a as u32 {
                let t0 = Instant::now();
                let n = perft(t, m, o, d, &mut buf);
                println!("perft {} {} ({:.2}s)", d, n, t0.elapsed().as_secs_f64());
            }
        }
        "positions" => {
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            for _ in 0..a {
                let (mut tiles, mut me, mut opp) = start();
                for _ply in 0..150 {
                    let (n, _) = bb::generate(tiles, me, opp, &mut buf);
                    let mut moves: Vec<i32> = buf[..n].to_vec();
                    moves.sort();
                    let list: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
                    println!("{{\"tiles\":{},\"me\":{},\"opp\":{},\"moves\":[{}]}}", tiles, me, opp, list.join(","));
                    if n == 0 {
                        std::mem::swap(&mut me, &mut opp);
                        continue;
                    }
                    let m = moves[rng.below(n)];
                    tiles = bb::apply(tiles, m);
                    let nm = bb::apply_pieces(me, m);
                    if bb::dest(m) == bb::GOAL {
                        break;
                    }
                    me = opp;
                    opp = nm;
                }
            }
        }
        "games" => {
            let b_max = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20usize);
            for a_ in 1..=a as usize {
                for b_ in 0..b_max {
                    let (mut tiles, p0, p1) = start();
                    let mut pieces = [p0, p1];
                    let mut counts: HashMap<(u64, u64, u64), u32> = HashMap::new();
                    *counts.entry((tiles, pieces[0], pieces[1])).or_insert(0) += 1;
                    let (mut winner, mut ply) = (-1i32, 0usize);
                    while ply < 150 {
                        let side = ply & 1;
                        let me = pieces[side];
                        let (n, _) = bb::generate(tiles, me, pieces[1 - side], &mut buf);
                        if n > 0 {
                            let m = buf[(ply * a_ + b_) % n];
                            tiles = bb::apply(tiles, m);
                            pieces[side] = bb::apply_pieces(me, m);
                            if bb::dest(m) == bb::GOAL {
                                winner = side as i32;
                                ply += 1;
                                break;
                            }
                        }
                        let next = 1 - side;
                        let seen = counts.entry((tiles, pieces[next], pieces[side])).or_insert(0);
                        *seen += 1;
                        if *seen >= 3 {
                            ply += 1;
                            break;
                        }
                        ply += 1;
                    }
                    println!("{} {} {} {}", a_, b_, winner, ply);
                }
            }
        }
        "playouts" => {
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            let t0 = Instant::now();
            let (mut plies, mut wins) = (0u64, 0u64);
            for _ in 0..a {
                let (mut tiles, mut me, mut opp) = start();
                for _ in 0..150 {
                    plies += 1;
                    let (n, _) = bb::generate(tiles, me, opp, &mut buf);
                    if n == 0 {
                        std::mem::swap(&mut me, &mut opp);
                        continue;
                    }
                    let m = buf[rng.below(n)];
                    tiles = bb::apply(tiles, m);
                    let nm = bb::apply_pieces(me, m);
                    if bb::dest(m) == bb::GOAL {
                        wins += 1;
                        break;
                    }
                    me = opp;
                    opp = nm;
                }
            }
            let s = t0.elapsed().as_secs_f64();
            println!("playouts {} plies {} wins {} {:.2}s -> {:.0} plies/s", a, plies, wins, s, plies as f64 / s);
        }
        _ => eprintln!("unknown mode"),
    }
}

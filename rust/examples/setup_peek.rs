//! 배치 하나를 그림으로 보여 주고, 처음 몇 수를 엔진으로 두며 판을 찍는다.
//!   cargo run --release --example setup_peek -- <tiles> <c1,c2,cx,cy> <수> <ms>
use slider::ai::{generate_turn, next_turn_pieces};
use slider::bitboard::{self as bb, GOAL, MAX_MOVES, SIZE};
use slider::setup::{engine, Setup};

fn draw(tiles: u64, p: &[u64; 2]) -> String {
    let mut s = String::new();
    for r in 0..SIZE {
        for c in 0..SIZE {
            let cell = r * SIZE + c;
            let b = 1u64 << cell;
            s.push(if p[0] & b != 0 { 'A' } else if p[1] & b != 0 { 'B' } else if tiles & b != 0 { '#' } else if cell == GOAL { '*' } else { '.' });
        }
        s.push('\n');
    }
    s
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let tiles: u64 = a[0].parse().unwrap();
    let c: Vec<usize> = a[1].split(',').map(|x| x.parse().unwrap()).collect();
    let plies: usize = a[2].parse().unwrap();
    let ms: u64 = a[3].parse().unwrap();
    let s = Setup { tiles, pieces: [[c[0], c[1]], [c[2], c[3]]], step: 16, owner: [0, 1], swap_used: false };
    print!("{}", s.diagram());
    let (mut t, mut p, mut cur) = s.start();
    let mut buf = vec![0i32; MAX_MOVES];
    for ply in 0..plies {
        let side = ply & 1;
        let (n, _) = generate_turn(t, p[side], p[1 - side], cur[side], &mut buf);
        if n == 0 {
            println!("{} 쪽{} 넘김", ply, side);
            cur[side] = next_turn_pieces(p[side], cur[side], p[side]);
            continue;
        }
        let r = engine(ms).search_turn(t, p[side], p[1 - side], cur[side], cur[1 - side]);
        t = bb::apply(t, r.mv);
        let after = bb::apply_pieces(p[side], r.mv);
        cur[side] = next_turn_pieces(p[side], cur[side], after);
        p[side] = after;
        println!("{} 쪽{} 수 {}개 점수 {} 깊이 {}", ply, side, n, r.score, r.depth);
        if ply % 10 == 9 || ply + 1 == plies {
            print!("{}", draw(t, &p));
        }
    }
}

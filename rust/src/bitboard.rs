//! 7x7 보드를 u64 비트보드로 다루는 유틸리티. 칸 번호 = row * 7 + col, 비트가 1이면 타일이 있는 칸.
//! 말(piece)도 u64 비트마스크이고, 말이 서 있는 칸은 항상 타일이다.
//!
//! 한 턴 = 내 말 하나를 골라, 타일 위를 원하는 만큼 걸은 뒤(다른 말이 있는 칸은 못 지나간다),
//! 빈 칸 방향으로 서 있는 타일째 미끄러지기. 미끄러짐은 다른 타일이나 벽 앞에서 멈춘다.
//! 이동은 i32 하나: from | dir << 6 | dest << 8 | src << 14
//! (src = 움직이는 말이 턴을 시작한 칸, from = 미끄러지기 시작한 타일, dest = 멈춘 칸)

pub const SIZE: usize = 7;
pub const CELLS: usize = SIZE * SIZE;
pub const GOAL: usize = 3 * SIZE + 3;
/// 말 2개 x 걸어갈 수 있는 타일 x 4방향. 넉넉하게 잡는다.
pub const MAX_MOVES: usize = CELLS * 4 * 2;

pub const SYMBOLS: [&str; 4] = ["w", "s", "a", "d"];
const DR: [i32; 4] = [-1, 1, 0, 0];
const DC: [i32; 4] = [0, 0, -1, 1];

const fn neighbor_table() -> [[i8; 4]; CELLS] {
    let mut table = [[-1i8; 4]; CELLS];
    let mut cell = 0;
    while cell < CELLS {
        let mut d = 0;
        while d < 4 {
            let r = (cell / SIZE) as i32 + DR[d];
            let c = (cell % SIZE) as i32 + DC[d];
            if r >= 0 && r < SIZE as i32 && c >= 0 && c < SIZE as i32 {
                table[cell][d] = (r * SIZE as i32 + c) as i8;
            }
            d += 1;
        }
        cell += 1;
    }
    table
}

pub static NEIGHBOR: [[i8; 4]; CELLS] = neighbor_table();

/// {row, col} 좌표 목록을 u64 비트마스크로 바꾼다.
pub fn pieces(locs: &[(usize, usize)]) -> u64 {
    locs.iter().fold(0u64, |mask, &(r, c)| mask | 1u64 << (r * SIZE + c))
}

/// 0/1 보드 배열을 타일 비트보드로 바꾼다.
pub fn from_array(board: &[[u8; SIZE]; SIZE]) -> u64 {
    let mut tiles = 0u64;
    for (r, row) in board.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v == 1 {
                tiles |= 1u64 << (r * SIZE + c);
            }
        }
    }
    tiles
}

#[inline(always)]
pub fn from(m: i32) -> usize {
    (m & 63) as usize
}

#[inline(always)]
pub fn dir(m: i32) -> usize {
    ((m >> 6) & 3) as usize
}

#[inline(always)]
pub fn dest(m: i32) -> usize {
    ((m >> 8) & 63) as usize
}

#[inline(always)]
pub fn src(m: i32) -> usize {
    ((m >> 14) & 63) as usize
}

/// 이동 후의 타일 비트보드.
#[inline(always)]
pub fn apply(tiles: u64, m: i32) -> u64 {
    (tiles & !(1u64 << from(m))) | (1u64 << dest(m))
}

/// 이동 후의 (움직인 쪽) 말 비트마스크.
#[inline(always)]
pub fn apply_pieces(pieces: u64, m: i32) -> u64 {
    (pieces & !(1u64 << src(m))) | (1u64 << dest(m))
}

#[inline(always)]
pub fn neighbor(cell: usize, d: usize) -> i32 {
    NEIGHBOR[cell][d] as i32
}

#[inline(always)]
pub fn is_tile(tiles: u64, cell: usize) -> bool {
    (tiles >> cell) & 1 != 0
}

/// start 에 있는 말이 걸어서 갈 수 있는 타일 칸들(start 포함)의 비트마스크. blocked 칸은 지나갈 수 없다.
pub fn reach(tiles: u64, start: usize, blocked: u64) -> u64 {
    let mut visited = 1u64 << start;
    let mut frontier = visited;
    while frontier != 0 {
        let cell = frontier.trailing_zeros() as usize;
        frontier &= frontier - 1;
        for d in 0..4 {
            let n = NEIGHBOR[cell][d];
            if n >= 0 {
                let n = n as usize;
                if is_tile(tiles, n) && !is_tile(visited, n) && !is_tile(blocked, n) {
                    visited |= 1u64 << n;
                    frontier |= 1u64 << n;
                }
            }
        }
    }
    visited
}

/// me 의 모든 말에 대해, 각 말이 걸어갈 수 있는 타일에서 미끄러질 수 있는 모든 이동을 out 에 채우고
/// (이동 개수, 모든 말이 걸을 수 있는 타일의 합집합) 을 돌려준다.
/// 둘 수 있는 수가 없으면 개수가 0 (이때는 차례를 넘긴다).
pub fn generate(tiles: u64, me: u64, opp: u64, out: &mut [i32]) -> (usize, u64) {
    let occupied = me | opp;
    let mut union = 0u64;
    let mut count = 0usize;
    let mut p = me;
    while p != 0 {
        let s = p.trailing_zeros() as usize;
        p &= p - 1;
        let r = reach(tiles, s, occupied & !(1u64 << s));
        union |= r;
        count = generate_from(tiles, r, s, out, count);
    }
    (count, union)
}

fn generate_from(tiles: u64, mut reach: u64, src: usize, out: &mut [i32], mut count: usize) -> usize {
    while reach != 0 {
        let cell = reach.trailing_zeros() as usize;
        reach &= reach - 1;
        for d in 0..4 {
            let n = NEIGHBOR[cell][d];
            if n < 0 || is_tile(tiles, n as usize) {
                continue;
            }
            let mut dest = n as usize;
            loop {
                let next = NEIGHBOR[dest][d];
                if next >= 0 && !is_tile(tiles, next as usize) {
                    dest = next as usize;
                } else {
                    break;
                }
            }
            out[count] = (cell | d << 6 | dest << 8 | src << 14) as i32;
            count += 1;
        }
    }
    count
}

/// 이동을 콘솔 게임이 실행할 수 있는 "w/s/a/d" 목록으로 바꾼다 (움직이는 말이 걷는 경로 + 마지막 미끄러짐).
pub fn to_symbols(tiles: u64, me: u64, opp: u64, m: i32) -> Vec<&'static str> {
    let start = src(m);
    let blocked = (me | opp) & !(1u64 << start);
    let mut parent = [-2i32; CELLS];
    let mut parent_dir = [0usize; CELLS];
    parent[start] = -1;
    let mut queue = [0usize; CELLS];
    let (mut head, mut tail) = (0, 1);
    queue[0] = start;
    while head < tail {
        let cell = queue[head];
        head += 1;
        for d in 0..4 {
            let n = NEIGHBOR[cell][d];
            if n >= 0 {
                let n = n as usize;
                if is_tile(tiles, n) && !is_tile(blocked, n) && parent[n] == -2 {
                    parent[n] = cell as i32;
                    parent_dir[n] = d;
                    queue[tail] = n;
                    tail += 1;
                }
            }
        }
    }
    let mut symbols = vec![SYMBOLS[dir(m)]];
    let mut cell = from(m);
    while parent[cell] >= 0 {
        symbols.push(SYMBOLS[parent_dir[cell]]);
        cell = parent[cell] as usize;
    }
    // 마지막 미끄러짐을 맨 뒤로 보내고 걷는 경로를 시작점부터 순서대로 놓는다
    let slide = symbols.remove(0);
    symbols.reverse();
    symbols.push(slide);
    symbols
}

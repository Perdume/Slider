package org.example.Learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 7x7 보드를 long 비트보드로 다루는 유틸리티.
 * 칸 번호 = row * 7 + col, 비트가 1이면 슬라이더(타일)가 있는 칸.
 * 말(piece)도 long 비트마스크로 표현한다. 한 편이 말을 몇 개 가져도 되지만 보통 2개다.
 * 말이 서 있는 칸은 항상 타일이다.
 *
 * 한 턴 = 내 말 하나를 골라, 타일 위를 원하는 만큼 걸은 뒤(다른 말이 있는 칸은 못 지나간다),
 * 빈 칸 방향으로 서 있는 타일째 미끄러지기. 미끄러짐은 다른 타일이나 벽 앞에서 멈춘다.
 * 이동은 int 하나로 표현한다: from | dir << 6 | dest << 8 | src << 14
 * (src = 움직이는 말이 턴을 시작한 칸, from = 미끄러지기 시작한 타일, dest = 멈춘 칸)
 */
public final class BitBoard {
    public static final int SIZE = 7;
    public static final int CELLS = SIZE * SIZE;
    public static final int GOAL = 3 * SIZE + 3;
    // 말 2개 x 걸어갈 수 있는 타일 x 4방향. 넉넉하게 잡는다.
    public static final int MAX_MOVES = CELLS * 4 * 2;

    static final String[] SYMBOLS = {"w", "s", "a", "d"};
    private static final int[] DR = {-1, 1, 0, 0};
    private static final int[] DC = {0, 0, -1, 1};
    private static final int[][] NEIGHBOR = new int[CELLS][4];

    static {
        for (int cell = 0; cell < CELLS; cell++) {
            for (int d = 0; d < 4; d++) {
                int r = cell / SIZE + DR[d];
                int c = cell % SIZE + DC[d];
                NEIGHBOR[cell][d] = (r >= 0 && r < SIZE && c >= 0 && c < SIZE) ? r * SIZE + c : -1;
            }
        }
    }

    private BitBoard() {
    }

    public static long fromArray(int[][] board) {
        long tiles = 0;
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                if (board[r][c] == 1) {
                    tiles |= 1L << (r * SIZE + c);
                }
            }
        }
        return tiles;
    }

    public static int cell(int[] loc) {
        return loc[0] * SIZE + loc[1];
    }

    public static int[] loc(int cell) {
        return new int[]{cell / SIZE, cell % SIZE};
    }

    /** {row, col} 좌표 목록을 말 비트마스크로 바꾼다. */
    public static long pieces(int[][] locs) {
        long mask = 0;
        for (int[] loc : locs) {
            mask |= 1L << cell(loc);
        }
        return mask;
    }

    static int from(int move) {
        return move & 63;
    }

    static int dir(int move) {
        return (move >> 6) & 3;
    }

    static int dest(int move) {
        return (move >> 8) & 63;
    }

    static int src(int move) {
        return (move >> 14) & 63;
    }

    /** 이동 후의 타일 비트보드. */
    static long apply(long tiles, int move) {
        return (tiles & ~(1L << from(move))) | (1L << dest(move));
    }

    /** 이동 후의 (움직인 쪽) 말 비트마스크. */
    static long applyPieces(long pieces, int move) {
        return (pieces & ~(1L << src(move))) | (1L << dest(move));
    }

    static int neighbor(int cell, int dir) {
        return NEIGHBOR[cell][dir];
    }

    static boolean isTile(long tiles, int cell) {
        return (tiles >>> cell & 1L) != 0;
    }

    /** start 에 있는 말이 걸어서 갈 수 있는 타일 칸들(start 포함)의 비트마스크. blocked 칸은 지나갈 수 없다. */
    static long reach(long tiles, int start, long blocked) {
        long visited = 1L << start;
        long frontier = visited;
        while (frontier != 0) {
            int cell = Long.numberOfTrailingZeros(frontier);
            frontier &= frontier - 1;
            for (int d = 0; d < 4; d++) {
                int n = NEIGHBOR[cell][d];
                if (n >= 0 && isTile(tiles, n) && !isTile(visited, n) && !isTile(blocked, n)) {
                    visited |= 1L << n;
                    frontier |= 1L << n;
                }
            }
        }
        return visited;
    }

    /** me 의 모든 말이 걸어서 갈 수 있는 타일 칸들의 합집합. 다른 말(내 말 포함)이 있는 칸은 지나갈 수 없다. */
    static long reachAll(long tiles, long me, long opp) {
        long occupied = me | opp;
        long union = 0;
        for (long p = me; p != 0; p &= p - 1) {
            int src = Long.numberOfTrailingZeros(p);
            union |= reach(tiles, src, occupied & ~(1L << src));
        }
        return union;
    }

    /**
     * me 의 모든 말에 대해, 각 말이 걸어갈 수 있는 타일에서 미끄러질 수 있는 모든 이동을 out 에 채우고 개수를 돌려준다.
     * 둘 수 있는 수가 없으면 0 (이때는 차례를 넘긴다).
     */
    static int generate(long tiles, long me, long opp, int[] out) {
        return generate(tiles, me, opp, out, null);
    }

    /** reachOut 이 null 이 아니면 reachOut[0] 에 모든 말이 걸을 수 있는 타일의 합집합을 돌려준다. */
    static int generate(long tiles, long me, long opp, int[] out, long[] reachOut) {
        long occupied = me | opp;
        long union = 0;
        int count = 0;
        for (long p = me; p != 0; p &= p - 1) {
            int src = Long.numberOfTrailingZeros(p);
            long reach = reach(tiles, src, occupied & ~(1L << src));
            union |= reach;
            count = generateFrom(tiles, reach, src, out, count);
        }
        if (reachOut != null) {
            reachOut[0] = union;
        }
        return count;
    }

    private static int generateFrom(long tiles, long reach, int src, int[] out, int count) {
        while (reach != 0) {
            int cell = Long.numberOfTrailingZeros(reach);
            reach &= reach - 1;
            for (int d = 0; d < 4; d++) {
                int n = NEIGHBOR[cell][d];
                if (n < 0 || isTile(tiles, n)) {
                    continue;
                }
                int dest = n;
                int next = NEIGHBOR[dest][d];
                while (next >= 0 && !isTile(tiles, next)) {
                    dest = next;
                    next = NEIGHBOR[dest][d];
                }
                out[count++] = cell | d << 6 | dest << 8 | src << 14;
            }
        }
        return count;
    }

    /** 이동을 Game 이 실행할 수 있는 "w/s/a/d" 목록으로 바꾼다 (움직이는 말이 걷는 경로 + 마지막 미끄러짐). */
    static List<String> toSymbols(long tiles, long me, long opp, int move) {
        int start = src(move);
        long blocked = (me | opp) & ~(1L << start);
        int[] parent = new int[CELLS];
        int[] parentDir = new int[CELLS];
        java.util.Arrays.fill(parent, -2);
        parent[start] = -1;
        int[] queue = new int[CELLS];
        int head = 0;
        int tail = 0;
        queue[tail++] = start;
        while (head < tail) {
            int cell = queue[head++];
            for (int d = 0; d < 4; d++) {
                int n = NEIGHBOR[cell][d];
                if (n >= 0 && isTile(tiles, n) && !isTile(blocked, n) && parent[n] == -2) {
                    parent[n] = cell;
                    parentDir[n] = d;
                    queue[tail++] = n;
                }
            }
        }
        List<String> symbols = new ArrayList<>();
        symbols.add(SYMBOLS[dir(move)]);
        for (int cell = from(move); parent[cell] >= 0; cell = parent[cell]) {
            symbols.add(SYMBOLS[parentDir[cell]]);
        }
        Collections.reverse(symbols);
        return symbols;
    }
}

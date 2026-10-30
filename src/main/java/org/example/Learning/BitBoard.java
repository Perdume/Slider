package org.example.Learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 7x7 보드를 long 비트보드로 다루는 유틸리티.
 * 칸 번호 = row * 7 + col, 비트가 1이면 슬라이더(타일)가 있는 칸.
 *
 * 한 턴 = 타일 위를 원하는 만큼 걸은 뒤(상대 칸 제외), 빈 칸 방향으로
 * 서 있는 타일째 미끄러지기. 미끄러짐은 다른 타일이나 벽 앞에서 멈춘다.
 * 이동은 int 하나로 표현한다: from | dir << 6 | dest << 8
 */
public final class BitBoard {
    public static final int SIZE = 7;
    public static final int CELLS = SIZE * SIZE;
    public static final int GOAL = 3 * SIZE + 3;
    public static final int MAX_MOVES = CELLS * 4;

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

    static int from(int move) {
        return move & 63;
    }

    static int dir(int move) {
        return (move >> 6) & 3;
    }

    static int dest(int move) {
        return move >> 8;
    }

    static long apply(long tiles, int move) {
        return (tiles & ~(1L << from(move))) | (1L << dest(move));
    }

    static int neighbor(int cell, int dir) {
        return NEIGHBOR[cell][dir];
    }

    static boolean isTile(long tiles, int cell) {
        return (tiles >>> cell & 1L) != 0;
    }

    /** me 가 걸어서 갈 수 있는 타일 칸들(me 포함)의 비트마스크. */
    static long reach(long tiles, int me, int opp) {
        long visited = 1L << me;
        long frontier = visited;
        while (frontier != 0) {
            int cell = Long.numberOfTrailingZeros(frontier);
            frontier &= frontier - 1;
            for (int d = 0; d < 4; d++) {
                int n = NEIGHBOR[cell][d];
                if (n >= 0 && n != opp && isTile(tiles, n) && !isTile(visited, n)) {
                    visited |= 1L << n;
                    frontier |= 1L << n;
                }
            }
        }
        return visited;
    }

    /** reach 마스크의 각 타일에서 미끄러질 수 있는 모든 이동을 out 에 채우고 개수를 돌려준다. */
    static int generate(long tiles, int me, int opp, int[] out) {
        return generate(tiles, reach(tiles, me, opp), out);
    }

    static int generate(long tiles, long reach, int[] out) {
        int count = 0;
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
                out[count++] = cell | d << 6 | dest << 8;
            }
        }
        return count;
    }

    /** 이동을 Game 이 실행할 수 있는 "w/s/a/d" 목록으로 바꾼다 (걷는 경로 + 마지막 미끄러짐). */
    static List<String> toSymbols(long tiles, int me, int opp, int move) {
        int[] parent = new int[CELLS];
        int[] parentDir = new int[CELLS];
        java.util.Arrays.fill(parent, -2);
        parent[me] = -1;
        int[] queue = new int[CELLS];
        int head = 0;
        int tail = 0;
        queue[tail++] = me;
        while (head < tail) {
            int cell = queue[head++];
            for (int d = 0; d < 4; d++) {
                int n = NEIGHBOR[cell][d];
                if (n >= 0 && n != opp && isTile(tiles, n) && parent[n] == -2) {
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

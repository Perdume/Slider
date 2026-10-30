package org.example.Learning;

import org.example.Move;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class SliderAITest {

    private static int[][] startBoard() {
        return new int[][]{
                {1, 1, 0, 0, 0, 1, 1},
                {1, 0, 0, 0, 0, 0, 1},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 0, 1},
                {1, 1, 0, 0, 0, 1, 1}
        };
    }

    private static int[][] boardWith(int[]... tiles) {
        int[][] board = new int[7][7];
        for (int[] t : tiles) {
            board[t[0]][t[1]] = 1;
        }
        return board;
    }

    private static int[][] toArray(long tiles) {
        int[][] board = new int[7][7];
        for (int cell = 0; cell < BitBoard.CELLS; cell++) {
            if (BitBoard.isTile(tiles, cell)) {
                board[cell / 7][cell % 7] = 1;
            }
        }
        return board;
    }

    private static int[] delta(String symbol) {
        switch (symbol) {
            case "w": return new int[]{0, -1};
            case "s": return new int[]{0, 1};
            case "a": return new int[]{-1, 0};
            default: return new int[]{1, 0};
        }
    }

    /** Game.EnemyTurn 과 같은 방식으로 이동 목록을 실행하고 최종 위치를 돌려준다. */
    private static int[] execute(int[][] board, int[] me, int[] opp, List<String> symbols) {
        Move mv = new Move();
        int[] loc = me.clone();
        for (int i = 0; i < symbols.size(); i++) {
            int[] first = loc.clone();
            int[] d = delta(symbols.get(i));
            int[] sol = mv.tryMove(d[0], d[1], loc, board, opp);
            assertNotNull(sol, "Move rejected " + symbols);
            loc = new int[]{sol[0], sol[1]};
            board[loc[0]][loc[1]] = 1;
            if (sol[2] == 1) {
                board[first[0]][first[1]] = 0;
                assertEquals(symbols.size() - 1, i, "Slid before the last step " + symbols);
                return loc;
            }
        }
        fail("Never slid " + symbols);
        return null;
    }

    /** Move.tryMove 만으로 한 턴에 가능한 모든 (출발 칸, 도착 칸) 결과를 구한다. */
    private static Set<Integer> outcomesByGameRules(int[][] board, int[] me, int[] opp) {
        Move mv = new Move();
        Set<Integer> outcomes = new HashSet<>();
        Set<Integer> visited = new HashSet<>();
        Deque<int[]> queue = new ArrayDeque<>();
        queue.add(me);
        visited.add(BitBoard.cell(me));
        while (!queue.isEmpty()) {
            int[] loc = queue.poll();
            for (String s : BitBoard.SYMBOLS) {
                int[] d = delta(s);
                int[] sol = mv.tryMove(d[0], d[1], loc, board, opp);
                if (sol == null) {
                    continue;
                }
                int to = sol[0] * 7 + sol[1];
                if (sol[2] == 1) {
                    outcomes.add(BitBoard.cell(loc) * 100 + to);
                } else if (visited.add(to)) {
                    queue.add(new int[]{sol[0], sol[1]});
                }
            }
        }
        return outcomes;
    }

    @Test
    void moveGenerationMatchesGameRules() {
        Random random = new Random(1);
        int[] buffer = new int[BitBoard.MAX_MOVES];
        for (int game = 0; game < 300; game++) {
            long tiles = BitBoard.fromArray(startBoard());
            int me = BitBoard.cell(new int[]{0, 6});
            int opp = BitBoard.cell(new int[]{6, 0});
            for (int ply = 0; ply < 80; ply++) {
                int n = BitBoard.generate(tiles, me, opp, buffer);
                Set<Integer> expected = outcomesByGameRules(toArray(tiles), BitBoard.loc(me), BitBoard.loc(opp));
                Set<Integer> actual = new HashSet<>();
                for (int i = 0; i < n; i++) {
                    int m = buffer[i];
                    actual.add(BitBoard.from(m) * 100 + BitBoard.dest(m));
                    int[][] board = toArray(tiles);
                    int[] end = execute(board, BitBoard.loc(me), BitBoard.loc(opp), BitBoard.toSymbols(tiles, me, opp, m));
                    assertEquals(BitBoard.dest(m), BitBoard.cell(end));
                    assertEquals(BitBoard.apply(tiles, m), BitBoard.fromArray(board));
                }
                assertEquals(expected, actual);
                assertTrue(n > 0);

                int m = buffer[random.nextInt(n)];
                if (BitBoard.dest(m) == BitBoard.GOAL) {
                    break;
                }
                tiles = BitBoard.apply(tiles, m);
                int next = opp;
                opp = BitBoard.dest(m);
                me = next;
            }
        }
    }

    @Test
    void takesImmediateWin() {
        int[][] board = boardWith(new int[]{3, 0}, new int[]{3, 4}, new int[]{0, 6}, new int[]{6, 6});
        Calculate cal = new Calculate(board, new int[]{0, 6}, new int[]{3, 0}, 500);
        assertEquals(List.of("d"), cal.findsolve());
        assertEquals(0, cal.getPlayerWinRate());
    }

    @Test
    void walksToWinningTile() {
        // AI (6,0) -> (5,0) -> (4,0) -> (3,0) 걸어서 이동 후 오른쪽으로 미끄러지면 골
        int[][] board = boardWith(new int[]{6, 0}, new int[]{5, 0}, new int[]{4, 0}, new int[]{3, 0},
                new int[]{3, 4}, new int[]{0, 6});
        Calculate cal = new Calculate(board, new int[]{0, 6}, new int[]{6, 0}, 500);
        List<String> moves = cal.findsolve();
        int[] end = execute(board, new int[]{6, 0}, new int[]{0, 6}, moves);
        assertArrayEquals(new int[]{3, 3}, end);
    }

    @Test
    void blocksPlayersWinningThreat() {
        // 플레이어 (0,3) 이 아래로 미끄러지면 (4,3) 타일에 막혀 골에 멈춘다
        int[][] board = boardWith(new int[]{0, 3}, new int[]{4, 3}, new int[]{5, 3}, new int[]{6, 3},
                new int[]{0, 0}, new int[]{6, 0}, new int[]{6, 6});
        int[] player = {0, 3};
        int[] ai = {6, 3};
        long tiles = BitBoard.fromArray(board);
        assertTrue(canWin(tiles, BitBoard.cell(player), BitBoard.cell(ai)));

        Calculate cal = new Calculate(board, player, ai, 500);
        int[] end = execute(board, ai, player, cal.findsolve());
        assertFalse(canWin(BitBoard.fromArray(board), BitBoard.cell(player), BitBoard.cell(end)));
    }

    @Test
    void beatsRandomPlayer() {
        Random random = new Random(7);
        int[] buffer = new int[BitBoard.MAX_MOVES];
        int aiWins = 0;
        int randomWins = 0;
        int games = 10;
        for (int game = 0; game < games; game++) {
            long tiles = BitBoard.fromArray(startBoard());
            int player = BitBoard.cell(new int[]{0, 6});
            int ai = BitBoard.cell(new int[]{6, 0});
            SliderAI engine = new SliderAI(60, 40);
            for (int turn = 0; turn < 60; turn++) {
                int n = BitBoard.generate(tiles, player, ai, buffer);
                int m = buffer[random.nextInt(n)];
                tiles = BitBoard.apply(tiles, m);
                player = BitBoard.dest(m);
                if (player == BitBoard.GOAL) {
                    randomWins++;
                    break;
                }
                int am = engine.search(tiles, ai, player).move;
                tiles = BitBoard.apply(tiles, am);
                ai = BitBoard.dest(am);
                if (ai == BitBoard.GOAL) {
                    aiWins++;
                    break;
                }
            }
        }
        assertEquals(0, randomWins);
        // 시간 제한 탐색이라 결과가 조금씩 달라질 수 있어 여유를 둔다 (보통 전승)
        assertTrue(aiWins >= games - 2, "AI won only " + aiWins + " of " + games);
    }

    private static boolean canWin(long tiles, int me, int opp) {
        int[] buffer = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles, me, opp, buffer);
        for (int i = 0; i < n; i++) {
            if (BitBoard.dest(buffer[i]) == BitBoard.GOAL) {
                return true;
            }
        }
        return false;
    }
}

package org.example.Learning;

import org.example.Move;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class SliderAITest {

    private static final int[][] PLAYER_START = {{0, 0}, {0, 6}};
    private static final int[][] AI_START = {{6, 0}, {6, 6}};

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

    private static int[][] locs(long mask) {
        List<int[]> list = new ArrayList<>();
        for (long p = mask; p != 0; p &= p - 1) {
            list.add(BitBoard.loc(Long.numberOfTrailingZeros(p)));
        }
        return list.toArray(new int[0][]);
    }

    /** mover 를 뺀 mine 의 나머지 말과 theirs 모두. */
    private static int[][] othersOf(int[][] mine, int[] mover, int[][] theirs) {
        List<int[]> others = new ArrayList<>();
        for (int[] m : mine) {
            if (!Arrays.equals(m, mover)) {
                others.add(m);
            }
        }
        others.addAll(Arrays.asList(theirs));
        return others.toArray(new int[0][]);
    }

    /** Game.EnemyTurn 과 같은 방식으로 이동 목록을 실행하고 최종 위치를 돌려준다. */
    private static int[] execute(int[][] board, int[] mover, int[][] others, List<String> symbols) {
        Move mv = new Move();
        int[] loc = mover.clone();
        for (int i = 0; i < symbols.size(); i++) {
            int[] first = loc.clone();
            int[] d = delta(symbols.get(i));
            int[] sol = mv.tryMove(d[0], d[1], loc, board, others);
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

    /** Move.tryMove 만으로 한 턴에 가능한 모든 (말, 출발 타일, 도착 칸) 결과를 구한다. */
    private static Set<Integer> outcomesByGameRules(int[][] board, int[][] mine, int[][] theirs) {
        Move mv = new Move();
        Set<Integer> outcomes = new HashSet<>();
        for (int[] piece : mine) {
            int src = BitBoard.cell(piece);
            int[][] others = othersOf(mine, piece, theirs);
            Set<Integer> visited = new HashSet<>();
            Deque<int[]> queue = new ArrayDeque<>();
            queue.add(piece);
            visited.add(src);
            while (!queue.isEmpty()) {
                int[] loc = queue.poll();
                for (String s : BitBoard.SYMBOLS) {
                    int[] d = delta(s);
                    int[] sol = mv.tryMove(d[0], d[1], loc, board, others);
                    if (sol == null) {
                        continue;
                    }
                    int to = sol[0] * 7 + sol[1];
                    if (sol[2] == 1) {
                        outcomes.add(src * 10000 + BitBoard.cell(loc) * 100 + to);
                    } else if (visited.add(to)) {
                        queue.add(new int[]{sol[0], sol[1]});
                    }
                }
            }
        }
        return outcomes;
    }

    private static boolean canWin(long tiles, long me, long opp) {
        int[] buffer = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles | me | opp, me, opp, buffer);
        for (int i = 0; i < n; i++) {
            if (BitBoard.dest(buffer[i]) == BitBoard.GOAL) {
                return true;
            }
        }
        return false;
    }

    @Test
    void moveGenerationMatchesGameRules() {
        Random random = new Random(1);
        int[] buffer = new int[BitBoard.MAX_MOVES];
        for (int game = 0; game < 200; game++) {
            long tiles = BitBoard.fromArray(startBoard());
            long me = BitBoard.pieces(PLAYER_START);
            long opp = BitBoard.pieces(AI_START);
            for (int ply = 0; ply < 80; ply++) {
                int n = BitBoard.generate(tiles, me, opp, buffer);
                int[][] mine = locs(me);
                int[][] theirs = locs(opp);
                Set<Integer> expected = outcomesByGameRules(toArray(tiles), mine, theirs);
                Set<Integer> actual = new HashSet<>();
                for (int i = 0; i < n; i++) {
                    int m = buffer[i];
                    actual.add(BitBoard.src(m) * 10000 + BitBoard.from(m) * 100 + BitBoard.dest(m));
                    int[][] board = toArray(tiles);
                    int[] mover = BitBoard.loc(BitBoard.src(m));
                    int[] end = execute(board, mover, othersOf(mine, mover, theirs), BitBoard.toSymbols(tiles, me, opp, m));
                    assertEquals(BitBoard.dest(m), BitBoard.cell(end));
                    assertEquals(BitBoard.apply(tiles, m), BitBoard.fromArray(board));
                }
                assertEquals(expected, actual);

                if (n == 0) {
                    // 둘 수 있는 수가 없으면 차례를 넘긴다
                    long swap = me;
                    me = opp;
                    opp = swap;
                    continue;
                }
                int m = buffer[random.nextInt(n)];
                if (BitBoard.dest(m) == BitBoard.GOAL) {
                    break;
                }
                long movedPieces = BitBoard.applyPieces(me, m);
                tiles = BitBoard.apply(tiles, m);
                me = opp;
                opp = movedPieces;
            }
        }
    }

    @Test
    void anyPieceReachingTheGoalWins() {
        // AI 의 두 번째 말 (3,0) 이 오른쪽으로 미끄러지면 (3,4) 타일 앞, 즉 골에 멈춘다
        int[][] board = boardWith(new int[]{3, 0}, new int[]{3, 4}, new int[]{0, 0}, new int[]{0, 6}, new int[]{6, 6});
        Calculate cal = new Calculate(board, PLAYER_START, new int[][]{{6, 6}, {3, 0}}, 500);
        assertEquals(List.of("d"), cal.findsolve());
        assertArrayEquals(new int[]{3, 0}, cal.getPieceLocation());
        assertEquals(0, cal.getPlayerWinRate());
    }

    @Test
    void singlePieceStillWorks() {
        int[][] board = boardWith(new int[]{3, 0}, new int[]{3, 4}, new int[]{0, 6}, new int[]{6, 6});
        Calculate cal = new Calculate(board, new int[]{0, 6}, new int[]{3, 0}, 500);
        assertEquals(List.of("d"), cal.findsolve());
        assertEquals(0, cal.getPlayerWinRate());
    }

    @Test
    void walksToWinningTile() {
        // AI 말 (6,0) -> (5,0) -> (4,0) -> (3,0) 걸어서 이동 후 오른쪽으로 미끄러지면 골
        int[][] board = boardWith(new int[]{6, 0}, new int[]{5, 0}, new int[]{4, 0}, new int[]{3, 0},
                new int[]{3, 4}, new int[]{0, 0}, new int[]{0, 6}, new int[]{6, 6});
        int[][] aiPieces = {{6, 0}, {6, 6}};
        Calculate cal = new Calculate(board, PLAYER_START, aiPieces, 500);
        List<String> moves = cal.findsolve();
        int[] mover = cal.getPieceLocation();
        assertArrayEquals(new int[]{6, 0}, mover);
        int[] end = execute(board, mover, othersOf(aiPieces, mover, PLAYER_START), moves);
        assertArrayEquals(new int[]{3, 3}, end);
    }

    @Test
    void blocksPlayersWinningThreat() {
        // 플레이어 말 (0,3) 이 아래로 미끄러지면 (4,3) 타일에 막혀 골에 멈춘다
        int[][] board = boardWith(new int[]{0, 3}, new int[]{4, 3}, new int[]{5, 3}, new int[]{6, 3},
                new int[]{0, 0}, new int[]{6, 6});
        int[][] player = {{0, 3}, {0, 0}};
        int[][] ai = {{6, 3}, {6, 6}};
        long tiles = BitBoard.fromArray(board);
        assertTrue(canWin(tiles, BitBoard.pieces(player), BitBoard.pieces(ai)));

        Calculate cal = new Calculate(board, player, ai, 500);
        List<String> moves = cal.findsolve();
        int[] mover = cal.getPieceLocation();
        int[] end = execute(board, mover, othersOf(ai, mover, player), moves);
        long aiAfter = BitBoard.pieces(ai) & ~(1L << BitBoard.cell(mover)) | 1L << BitBoard.cell(end);
        assertFalse(canWin(BitBoard.fromArray(board), BitBoard.pieces(player), aiAfter));
    }

    @Test
    void passesWhenNoPieceCanMove() {
        // AI 말 (6,0) 은 두 이웃 칸이 모두 플레이어 말이라 걷지도 미끄러지지도 못한다
        int[][] board = boardWith(new int[]{6, 0}, new int[]{5, 0}, new int[]{6, 1});
        int[][] player = {{5, 0}, {6, 1}};
        int[][] ai = {{6, 0}};
        assertArrayEquals(new boolean[]{false}, Calculate.movablePieces(board, ai, player));

        Calculate cal = new Calculate(board, player, ai, 200);
        assertTrue(cal.findsolve().isEmpty());
        assertNull(cal.getPieceLocation());
    }

    @Test
    void movablePiecesReportsEachPiece() {
        int[][] board = boardWith(new int[]{6, 0}, new int[]{5, 0}, new int[]{6, 1}, new int[]{0, 6});
        int[][] mine = {{6, 0}, {0, 6}};
        int[][] theirs = {{5, 0}, {6, 1}};
        assertArrayEquals(new boolean[]{false, true}, Calculate.movablePieces(board, mine, theirs));
    }

    @Test
    void searchesNormallyWhenOpponentMustPass() {
        // 플레이어 말 (6,0) 은 AI 말에 둘러싸여 못 움직인다. 탐색이 차례 넘기기를 처리해야 한다.
        int[][] board = boardWith(new int[]{6, 0}, new int[]{5, 0}, new int[]{6, 1});
        int[][] player = {{6, 0}};
        int[][] ai = {{5, 0}, {6, 1}};
        Calculate cal = new Calculate(board, player, ai, 150);
        assertFalse(cal.findsolve().isEmpty());
        assertNotNull(cal.getPieceLocation());
    }

    @Test
    void thirdRepetitionIsADraw() {
        long tiles = BitBoard.fromArray(startBoard());
        long me = BitBoard.pieces(AI_START);
        long opp = BitBoard.pieces(PLAYER_START);
        int[] moves = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles, me, opp, moves);
        assertTrue(n > 0);

        // 모든 수가 "이미 2번 나온 국면"으로 이어지면, 어느 수를 둬도 3번째 반복이라 무승부다
        Map<Long, Integer> history = new HashMap<>();
        for (int i = 0; i < n; i++) {
            long key = SliderAI.positionHash(BitBoard.apply(tiles, moves[i]), opp, BitBoard.applyPieces(me, moves[i]));
            history.put(key, 2);
        }
        SliderAI ai = new SliderAI(100, 40);
        ai.setGameHistory(history);
        SliderAI.Result result = ai.search(tiles, me, opp);
        assertEquals(SliderAI.DRAW, result.score);
        assertTrue(result.move >= 0);
    }

    @Test
    void positionHashDistinguishesSideToMove() {
        long tiles = BitBoard.fromArray(startBoard());
        long a = BitBoard.pieces(PLAYER_START);
        long b = BitBoard.pieces(AI_START);
        assertNotEquals(SliderAI.positionHash(tiles, a, b), SliderAI.positionHash(tiles, b, a));
        assertEquals(SliderAI.positionHash(tiles, a, b), SliderAI.positionHash(tiles, a, b));
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
            long player = BitBoard.pieces(PLAYER_START);
            long ai = BitBoard.pieces(AI_START);
            SliderAI engine = new SliderAI(60, 40);
            for (int turn = 0; turn < 60; turn++) {
                int n = BitBoard.generate(tiles, player, ai, buffer);
                if (n > 0) {
                    int m = buffer[random.nextInt(n)];
                    tiles = BitBoard.apply(tiles, m);
                    player = BitBoard.applyPieces(player, m);
                    if (BitBoard.dest(m) == BitBoard.GOAL) {
                        randomWins++;
                        break;
                    }
                }
                int am = engine.search(tiles, ai, player).move;
                if (am >= 0) {
                    tiles = BitBoard.apply(tiles, am);
                    ai = BitBoard.applyPieces(ai, am);
                    if (BitBoard.dest(am) == BitBoard.GOAL) {
                        aiWins++;
                        break;
                    }
                }
            }
        }
        assertEquals(0, randomWins);
        // 시간 제한 탐색이라 결과가 조금씩 달라질 수 있어 여유를 둔다 (보통 전승)
        assertTrue(aiWins >= games - 2, "AI won only " + aiWins + " of " + games);
    }

    @Test
    void parallelSearchWithLmrReturnsALegalMoveAndFindsAnImmediateWin() {
        long tiles = BitBoard.fromArray(startBoard());
        long me = BitBoard.pieces(AI_START);
        long opp = BitBoard.pieces(PLAYER_START);
        int[] buffer = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles, me, opp, buffer);
        Set<Integer> legal = new HashSet<>();
        for (int i = 0; i < n; i++) {
            legal.add(buffer[i]);
        }
        for (int threads : new int[]{1, 4}) {
            SliderAI ai = new SliderAI(200, 40);
            ai.setOptions(true, true);
            ai.setThreads(threads);
            SliderAI.Result r = ai.search(tiles, me, opp);
            assertTrue(legal.contains(r.move), "스레드 " + threads + ": 둘 수 없는 수를 골랐습니다");
            assertTrue(r.depth >= 2, "스레드 " + threads + ": 깊이 " + r.depth);
        }
        // 바로 이길 수 있으면 병렬 탐색이어도 그 수를 둔다: (3,6) 의 말이 왼쪽으로 미끄러지면 골 (3,3) 왼쪽의 받침 (3,2) 앞, 즉 골에서 멈춘다
        long t2 = BitBoard.fromArray(boardWith(new int[]{3, 2}, new int[]{3, 6}, new int[]{0, 0}, new int[]{6, 6}));
        long me2 = BitBoard.pieces(new int[][]{{3, 6}});
        long opp2 = BitBoard.pieces(new int[][]{{0, 0}});
        SliderAI ai = new SliderAI(200, 40);
        ai.setOptions(true, true);
        ai.setThreads(4);
        SliderAI.Result r = ai.search(t2, me2, opp2);
        assertEquals(BitBoard.GOAL, BitBoard.dest(r.move));
    }
}

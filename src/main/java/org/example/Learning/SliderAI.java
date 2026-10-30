package org.example.Learning;

import java.util.Arrays;

/**
 * 반복 심화(iterative deepening) + 알파베타 가지치기 negamax 탐색 AI.
 * 주어진 시간 안에서 가능한 한 깊이 탐색하고, 트랜스포지션 테이블과
 * 킬러/히스토리 휴리스틱으로 수 정렬을 개선한다.
 * 점수는 항상 "지금 둘 차례인 쪽" 기준이다.
 *
 * 이 게임은 서로 위협을 막기만 하면 잘 끝나지 않아서, 순수 minimax 는 지지는 않지만 이기려 하지도 않는다.
 * 그래서 안전한 수들 중에서는 "상대가 직접적인 위협에만 반응한다"고 가정한 탐색으로 공격 계획을 세운다.
 */
public class SliderAI {
    public static final int WIN = 100_000;
    public static final int WIN_BOUND = WIN - 1_000;
    // 상대가 계획을 방해하지 않는다고 가정했을 때 찾은 승리 (확정 승리는 아니다)
    public static final int PLAN_WIN = 5_000;
    public static final int PLAN_BOUND = PLAN_WIN - 1_000;
    private static final int INF = WIN + 1;

    private static final int MAX_PLY = 64;
    private static final int NO_PLAN = Integer.MAX_VALUE;
    private static final int TT_SIZE = 1 << 19;
    private static final byte EXACT = 0, LOWER = 1, UPPER = 2;

    // 최선수와 이 점수 차이 이내면 안전한 수로 보고 공격 계획/함정으로 다시 고른다
    private static final int MARGIN = 40;
    // 상대 응수 중 내가 바로 이기게 되는 비율에 곱하는 보너스
    private static final int TRAP_WEIGHT = 60;
    // 이 ply 부터 상대는 직접적인 위협이 없으면 수를 쉰다고 가정한다 (1, 3 ply 의 상대 응수는 모두 본다)
    private static final int PLAN_PLY = 5;
    private static final int MOBILITY_WEIGHT = 1;
    private static final int REACH_WEIGHT = 2;
    // 이기기까지 남은 수(어림)별 점수: 2수 -> 150, 3수 -> 70, ...
    private static final int[] PLAN_VALUE = {0, 0, 150, 70, 30, 12};

    // 중앙에 가까울수록 높은 위치 점수
    private static final int[] POSITION = {
            0, 1, 2, 3, 2, 1, 0,
            1, 3, 4, 5, 4, 3, 1,
            2, 4, 6, 7, 6, 4, 2,
            3, 5, 7, 0, 7, 5, 3,
            2, 4, 6, 7, 6, 4, 2,
            1, 3, 4, 5, 4, 3, 1,
            0, 1, 2, 3, 2, 1, 0
    };

    // RAY[cell] = 골에서 cell 쪽으로 가는 방향 (골과 같은 행/열이 아니면 -1), BETWEEN[cell] = 골과 cell 사이 칸들
    private static final int[] RAY = new int[BitBoard.CELLS];
    // BACKSTOP[r] = 골에서 r 쪽 직선으로 들어오는 타일을 골에 멈추게 하는 칸
    private static final int[] BACKSTOP = new int[4];
    private static final int MAX_COST = 6;
    private static final long FULL = (1L << BitBoard.CELLS) - 1;
    private static final long GOAL_BIT = 1L << BitBoard.GOAL;
    private static final long COL_FIRST, COL_LAST, BORDER;
    private static final long[] BETWEEN = new long[BitBoard.CELLS];

    static {
        long first = 0, last = 0, border = 0;
        for (int cell = 0; cell < BitBoard.CELLS; cell++) {
            int r = cell / BitBoard.SIZE, c = cell % BitBoard.SIZE;
            if (c == 0) first |= 1L << cell;
            if (c == BitBoard.SIZE - 1) last |= 1L << cell;
            if (r == 0 || c == 0 || r == BitBoard.SIZE - 1 || c == BitBoard.SIZE - 1) border |= 1L << cell;
        }
        COL_FIRST = first;
        COL_LAST = last;
        BORDER = border;
        for (int r = 0; r < 4; r++) {
            BACKSTOP[r] = BitBoard.neighbor(BitBoard.GOAL, r ^ 1);
        }
        Arrays.fill(RAY, -1);
        for (int r = 0; r < 4; r++) {
            long between = 0;
            for (int cell = BitBoard.neighbor(BitBoard.GOAL, r); cell >= 0; cell = BitBoard.neighbor(cell, r)) {
                RAY[cell] = r;
                BETWEEN[cell] = between;
                between |= 1L << cell;
            }
        }
    }

    public static final class Result {
        public final int move;
        public final int score;
        public final int depth;
        public final long nodes;

        Result(int move, int score, int depth, long nodes) {
            this.move = move;
            this.score = score;
            this.depth = depth;
            this.nodes = nodes;
        }
    }

    private final long timeLimitMs;
    private final int maxDepth;

    private final long[] ttTiles = new long[TT_SIZE];
    private final int[] ttPos = new int[TT_SIZE];
    private final int[] ttScore = new int[TT_SIZE];
    private final int[] ttMove = new int[TT_SIZE];
    private final byte[] ttDepth = new byte[TT_SIZE];
    private final byte[] ttFlag = new byte[TT_SIZE];

    private final int[][] moves = new int[MAX_PLY][BitBoard.MAX_MOVES];
    private final int[][] keys = new int[MAX_PLY][BitBoard.MAX_MOVES];
    private final int[][] killers = new int[MAX_PLY][2];
    private final int[] history = new int[256];
    private final int[] scratch = new int[BitBoard.MAX_MOVES];
    private final int[] backCost = new int[4];

    private long deadline;
    private long nodes;
    private boolean aborted;
    private boolean mayAbort;
    private int optPly;

    public SliderAI(long timeLimitMs, int maxDepth) {
        this.timeLimitMs = timeLimitMs;
        this.maxDepth = Math.min(maxDepth, MAX_PLY - 2);
    }

    /**
     * me 차례에서 최선의 수를 찾는다. 둘 수 있는 수가 없으면 move 는 -1.
     * 1단계: 정확한 minimax 로 안전한 수(최선과 MARGIN 이내)를 고른다.
     * 2단계: 안전한 수들만 "상대는 직접적인 위협에만 반응한다"고 가정한 탐색으로 다시 비교해
     *        여러 수에 걸친 공격 계획이 있는 수를 고른다.
     */
    public Result search(long tiles, int me, int opp) {
        tiles |= 1L << me | 1L << opp;
        long start = System.currentTimeMillis();
        nodes = 0;

        int[] rootMoves = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles, me, opp, rootMoves);
        if (n == 0) {
            return new Result(-1, 0, 0, 0);
        }
        for (int i = 0; i < n; i++) {
            if (BitBoard.dest(rootMoves[i]) == BitBoard.GOAL) {
                return new Result(rootMoves[i], WIN, 1, 1);
            }
        }
        rootMoves = Arrays.copyOf(rootMoves, n);
        int[] scores = new int[n];
        optPly = NO_PLAN;
        int depth = iterate(tiles, me, opp, rootMoves, scores, start + timeLimitMs / 2);
        int best = scores[0];
        int safe = 0;
        while (safe < n && scores[safe] > best - MARGIN) {
            safe++;
        }
        int move = rootMoves[0];
        if (Math.abs(best) < WIN_BOUND && safe > 1) {
            int[] planMoves = Arrays.copyOf(rootMoves, safe);
            int[] planScores = new int[safe];
            optPly = PLAN_PLY;
            iterate(tiles, me, opp, planMoves, planScores, start + timeLimitMs);
            optPly = NO_PLAN;
            move = choose(tiles, me, opp, planMoves, planScores);
        }
        return new Result(move, best, depth, nodes);
    }

    /** 반복 심화로 rootMoves 를 탐색하고 마지막으로 끝낸 깊이의 점수 순서로 정렬한다. 끝낸 깊이를 돌려준다. */
    private int iterate(long tiles, int me, int opp, int[] rootMoves, int[] scores, long until) {
        deadline = until;
        aborted = false;
        int n = rootMoves.length;
        int[] work = rootMoves.clone();
        int[] workScores = new int[n];
        int done = 0;
        for (int depth = 1; depth <= maxDepth; depth++) {
            for (int i = 0; i < history.length; i++) {
                history[i] >>= 1;
            }
            // 깊이 1은 시간이 끝나도 끝까지 탐색해서 항상 결과가 있게 한다
            mayAbort = depth > 1;
            int best = -INF;
            for (int i = 0; i < n && !aborted; i++) {
                int m = work[i];
                // 최선수와 MARGIN 이내인 수는 정확한 점수를 얻도록 창을 넓힌다
                int floor = best == -INF ? -INF : best - MARGIN;
                workScores[i] = -negamax(BitBoard.apply(tiles, m), opp, BitBoard.dest(m), depth - 1, -INF, -floor, 1);
                best = Math.max(best, workScores[i]);
            }
            if (aborted) {
                break;
            }
            sortByScore(work, workScores, n);
            System.arraycopy(work, 0, rootMoves, 0, n);
            System.arraycopy(workScores, 0, scores, 0, n);
            done = depth;
            if (Math.abs(best) >= PLAN_BOUND || System.currentTimeMillis() > deadline) {
                break;
            }
        }
        return done;
    }

    /**
     * 최선수와 점수 차이가 MARGIN 미만인 안전한 수들 중에서
     * 상대가 막지 못하면 바로 이길 수 있는 응수의 비율(함정)이 높은 수를 우선한다.
     * 완벽한 상대에게는 비슷한 결과지만, 실수하는 상대(사람)를 이기는 데 도움이 된다.
     */
    private int choose(long tiles, int me, int opp, int[] rootMoves, int[] scores) {
        int bestScore = scores[0];
        if (Math.abs(bestScore) >= PLAN_BOUND) {
            return rootMoves[0];
        }
        int[] replies = new int[BitBoard.MAX_MOVES];
        int chosen = rootMoves[0];
        double chosenKey = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < rootMoves.length && scores[i] > bestScore - MARGIN; i++) {
            int m = rootMoves[i];
            long after = BitBoard.apply(tiles, m);
            int myCell = BitBoard.dest(m);
            int n = BitBoard.generate(after, opp, myCell, replies);
            int traps = 0;
            for (int j = 0; j < n; j++) {
                int r = replies[j];
                if (canWin(BitBoard.apply(after, r), myCell, BitBoard.dest(r))) {
                    traps++;
                }
            }
            double key = scores[i] + (n == 0 ? 0 : TRAP_WEIGHT * traps / (double) n);
            if (key > chosenKey) {
                chosenKey = key;
                chosen = m;
            }
        }
        return chosen;
    }

    private boolean canWin(long tiles, int me, int opp) {
        int n = BitBoard.generate(tiles, me, opp, scratch);
        for (int i = 0; i < n; i++) {
            if (BitBoard.dest(scratch[i]) == BitBoard.GOAL) {
                return true;
            }
        }
        return false;
    }

    private int negamax(long tiles, int me, int opp, int depth, int alpha, int beta, int ply) {
        if ((++nodes & 1023) == 0 && mayAbort && System.currentTimeMillis() > deadline) {
            aborted = true;
        }
        if (aborted) {
            return 0;
        }

        // 같은 국면이라도 탐색 방식(정확/낙관, 낙관까지 남은 수)이 다르면 값이 다르므로 TT 키에 넣는다
        boolean optimistic = ply >= optPly;
        int mode = optimistic ? 1 + (ply & 1) : optPly == NO_PLAN ? 0 : 3 + optPly - ply;
        int idx = index(tiles, me, opp, mode);
        int pos = ((me << 8 | opp) << 4 | mode) + 1;
        int ttBest = -1;
        if (ttPos[idx] == pos && ttTiles[idx] == tiles) {
            ttBest = ttMove[idx];
            if (ttDepth[idx] >= depth) {
                int s = fromTT(ttScore[idx], ply);
                byte flag = ttFlag[idx];
                if (flag == EXACT || (flag == LOWER && s >= beta) || (flag == UPPER && s <= alpha)) {
                    return s;
                }
            }
        }

        int[] list = moves[ply];
        long reach = BitBoard.reach(tiles, me, opp);
        int n = BitBoard.generate(tiles, reach, list);
        for (int i = 0; i < n; i++) {
            if (BitBoard.dest(list[i]) == BitBoard.GOAL) {
                return (optimistic ? PLAN_WIN : WIN) - ply;
            }
        }
        if (depth <= 0 || ply >= MAX_PLY - 1) {
            return evaluate(tiles, me, opp, reach, list, n);
        }
        if (optimistic && (ply & 1) == 1 && !canWin(tiles, opp, me)) {
            // 깊은 곳에서 상대는 바로 이기거나 직접적인 위협을 막을 때만 움직인다고 가정한다.
            // 이렇게 해야 AI 가 여러 수에 걸친 공격 계획을 세운다.
            return -negamax(tiles, opp, me, depth - 1, -beta, -alpha, ply + 1);
        }
        if (n == 0) {
            // 움직일 곳이 없으면 차례를 넘긴다 (기본 보드에서는 일어나지 않는다)
            return -negamax(tiles, opp, me, depth - 1, -beta, -alpha, ply + 1);
        }

        int[] key = keys[ply];
        for (int i = 0; i < n; i++) {
            int m = list[i];
            if (m == ttBest) {
                key[i] = 1 << 30;
            } else if (m == killers[ply][0]) {
                key[i] = 1 << 29;
            } else if (m == killers[ply][1]) {
                key[i] = 1 << 28;
            } else {
                key[i] = history[m & 255] + POSITION[BitBoard.dest(m)];
            }
        }

        int origAlpha = alpha;
        int best = -INF;
        int bestMove = -1;
        for (int i = 0; i < n; i++) {
            pickNext(list, key, i, n);
            int m = list[i];
            int s = -negamax(BitBoard.apply(tiles, m), opp, BitBoard.dest(m), depth - 1, -beta, -alpha, ply + 1);
            if (aborted) {
                return 0;
            }
            if (s > best) {
                best = s;
                bestMove = m;
            }
            if (s > alpha) {
                alpha = s;
            }
            if (alpha >= beta) {
                if (m != killers[ply][0]) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = m;
                }
                history[m & 255] = Math.min(history[m & 255] + depth * depth, 1 << 20);
                break;
            }
        }

        ttTiles[idx] = tiles;
        ttPos[idx] = pos;
        ttScore[idx] = toTT(best, ply);
        ttMove[idx] = bestMove;
        ttDepth[idx] = (byte) depth;
        ttFlag[idx] = best <= origAlpha ? UPPER : best >= beta ? LOWER : EXACT;
        return best;
    }

    /** 말단 평가. me 가 둘 차례이고, me 가 바로 이길 수 없는 상태라고 가정한다. */
    private int evaluate(long tiles, int me, int opp, long reachMe, int[] listMe, int movesMe) {
        long reachOpp = BitBoard.reach(tiles, opp, me);
        int movesOpp = BitBoard.generate(tiles, reachOpp, scratch);
        int threatsOpp = 0;
        for (int i = 0; i < movesOpp; i++) {
            if (BitBoard.dest(scratch[i]) == BitBoard.GOAL) {
                threatsOpp++;
            }
        }

        int score = 0;
        if (threatsOpp == 1) {
            score -= 150; // 막을 수는 있지만 한 수를 강제로 써야 한다
        } else if (threatsOpp > 1) {
            score -= 400 + 50 * threatsOpp;
        }
        score += MOBILITY_WEIGHT * (movesMe - movesOpp);
        score += REACH_WEIGHT * (Long.bitCount(reachMe) - Long.bitCount(reachOpp));
        backstopCosts(tiles, backCost);
        score += planValue(tiles, reachMe, listMe, movesMe, backCost) - planValue(tiles, reachOpp, scratch, movesOpp, backCost);
        score += POSITION[me] - POSITION[opp];
        return score;
    }

    /**
     * 받침 타일 비용: 골 바로 옆 칸 (골에서 r 반대 방향) 에 타일을 세우려면 몇 개의 타일을 더 옮겨야 하는지.
     * 타일은 다른 타일이나 벽 앞에서만 멈추므로, 타일/벽에서 빈 칸을 따라 잰 거리로 어림한다.
     */
    private static void backstopCosts(long tiles, int[] out) {
        long solid = tiles;
        long layer = (spread(tiles) | BORDER) & ~tiles & ~GOAL_BIT;
        Arrays.fill(out, MAX_COST);
        for (int cost = 1; cost < MAX_COST && layer != 0; cost++) {
            for (int r = 0; r < 4; r++) {
                if (out[r] == MAX_COST && (layer >>> BACKSTOP[r] & 1) != 0) {
                    out[r] = cost;
                }
            }
            solid |= layer;
            layer = spread(layer) & ~solid & ~GOAL_BIT & FULL;
        }
        for (int r = 0; r < 4; r++) {
            if (BitBoard.isTile(tiles, BACKSTOP[r])) {
                out[r] = 0;
            }
        }
    }

    private static long spread(long m) {
        return ((m << 7) | (m >>> 7) | ((m & ~COL_LAST) << 1) | ((m & ~COL_FIRST) >>> 1)) & FULL;
    }

    /**
     * 각 직선(골에서 바깥 방향 r)으로 이기기까지 남은 수를 어림해서 점수로 바꾼다.
     * 비용 = 받침 타일 비용 + 직선 위 첫 타일을 밟는 비용(0: 이미 밟을 수 있음, 1: 다음 수에 올라감, 2: 그 외) + 1
     */
    private static int planValue(long tiles, long reach, int[] list, int n, int[] backCost) {
        int lineNext = 0;
        for (int i = 0; i < n; i++) {
            int m = list[i];
            int dest = BitBoard.dest(m);
            int r = RAY[dest];
            if (r >= 0 && ((tiles & ~(1L << BitBoard.from(m))) & BETWEEN[dest]) == 0) {
                lineNext |= 1 << r;
            }
        }
        int score = 0;
        for (int r = 0; r < 4; r++) {
            int cell = BitBoard.neighbor(BitBoard.GOAL, r);
            while (cell >= 0 && !BitBoard.isTile(tiles, cell)) {
                cell = BitBoard.neighbor(cell, r);
            }
            int lineCost = cell >= 0 && BitBoard.isTile(reach, cell) ? 0 : (lineNext >> r & 1) != 0 ? 1 : 2;
            int cost = backCost[r] + lineCost + 1;
            if (cost < PLAN_VALUE.length) {
                score += PLAN_VALUE[cost];
            }
        }
        return score;
    }

    private static void pickNext(int[] list, int[] key, int from, int n) {
        int best = from;
        for (int j = from + 1; j < n; j++) {
            if (key[j] > key[best]) {
                best = j;
            }
        }
        if (best != from) {
            int m = list[from];
            list[from] = list[best];
            list[best] = m;
            int k = key[from];
            key[from] = key[best];
            key[best] = k;
        }
    }

    private static void sortByScore(int[] list, int[] scores, int n) {
        // 삽입 정렬 (안정 정렬이라 동점이면 이전 순서를 유지한다)
        for (int i = 1; i < n; i++) {
            int m = list[i];
            int s = scores[i];
            int j = i - 1;
            while (j >= 0 && scores[j] < s) {
                list[j + 1] = list[j];
                scores[j + 1] = scores[j];
                j--;
            }
            list[j + 1] = m;
            scores[j + 1] = s;
        }
    }

    private static int index(long tiles, int me, int opp, int mode) {
        long h = tiles * 0x9E3779B97F4A7C15L ^ ((me * 64L + opp) * 16 + mode) * 0xC2B2AE3D27D4EB4FL;
        h ^= h >>> 31;
        h *= 0xBF58476D1CE4E5B9L;
        h ^= h >>> 29;
        return (int) h & (TT_SIZE - 1);
    }

    private static int toTT(int score, int ply) {
        if (score >= PLAN_BOUND) {
            return score + ply;
        }
        if (score <= -PLAN_BOUND) {
            return score - ply;
        }
        return score;
    }

    private static int fromTT(int score, int ply) {
        if (score >= PLAN_BOUND) {
            return score - ply;
        }
        if (score <= -PLAN_BOUND) {
            return score + ply;
        }
        return score;
    }
}

package org.example.Learning;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Rust 포팅 검증용 기준 데이터(rust/tests/golden.txt)를 Java 구현에서 뽑는 도구.
 * 무작위로 둔 국면마다 이동 생성 순서, 평가 특징, 정해진 깊이의 탐색 결과(수, 점수, 끝낸 깊이)를 기록한다.
 * 탐색 시간 제한은 사실상 없애서(10분) 깊이로만 끝나게 하므로 결과가 실행마다 같다.
 *
 * 만드는 법 (저장소 루트에서):
 *   mvn -q compile
 *   javac -cp target/classes -d /tmp/goldengen rust/tools/GoldenGen.java
 *   java -cp target/classes:/tmp/goldengen org.example.Learning.GoldenGen rust/tests/golden.txt
 *
 * 한 줄 형식 (공백으로 구분):
 *   tiles me opp depth nHist {hash count}*nHist nMoves {move}*nMoves bestMove score reachedDepth {feature}*11
 */
public final class GoldenGen {
    private GoldenGen() {
    }

    public static void main(String[] args) throws Exception {
        String out = args.length > 0 ? args[0] : "golden.txt";
        Random rnd = new Random(20261102L);
        long startTiles = BitBoard.fromArray(Arena.START_BOARD);
        long start0 = BitBoard.pieces(Arena.SIDE0_START);
        long start1 = BitBoard.pieces(Arena.SIDE1_START);
        int[] depths = {3, 4, 5, 6};
        int cases = 0;
        int passes = 0;

        try (PrintWriter w = new PrintWriter(out, StandardCharsets.UTF_8)) {
            while (cases < 80) {
                long tiles = startTiles;
                long[] pieces = {start0, start1};
                int plies = rnd.nextInt(80);
                int side = 0;
                boolean ended = false;
                int[] buf = new int[BitBoard.MAX_MOVES];
                for (int p = 0; p < plies; p++) {
                    side = p & 1;
                    int n = BitBoard.generate(tiles, pieces[side], pieces[1 - side], buf);
                    if (n == 0) {
                        continue;
                    }
                    int m = buf[rnd.nextInt(n)];
                    tiles = BitBoard.apply(tiles, m);
                    pieces[side] = BitBoard.applyPieces(pieces[side], m);
                    if (BitBoard.dest(m) == BitBoard.GOAL) {
                        ended = true;
                        break;
                    }
                }
                if (ended) {
                    continue;
                }
                side = plies & 1;
                long me = pieces[side];
                long opp = pieces[1 - side];
                int n = BitBoard.generate(tiles, me, opp, buf);
                if (n == 0) {
                    passes++;
                }

                Map<Long, Integer> history = new HashMap<>();
                if (cases % 2 == 1) {
                    // 절반 정도는 "이미 두 번 나온 국면"을 심어서 실제 게임의 반복 이력 경로도 검증한다
                    for (int i = 0; i < Math.min(n, 3); i++) {
                        int m = buf[rnd.nextInt(n)];
                        history.put(SliderAI.positionHash(BitBoard.apply(tiles, m), opp, BitBoard.applyPieces(me, m)), 2);
                    }
                }
                emit(w, tiles, me, opp, depths[cases % depths.length], history);
                cases++;
            }

            // 손으로 만든 국면: 갇혀서 차례를 넘기는 경우, 상대가 한 수로 이기는 위협이 있는 경우
            int[][] stuck = {
                    {1, 1, 0, 0, 0, 0, 0},
                    {1, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 1, 1}
            };
            long stuckTiles = BitBoard.fromArray(stuck);
            // me 의 유일한 말 (0,0) 은 이웃 타일 (0,1), (1,0) 을 상대 말이 막고 있어서 걸을 수도 미끄러질 수도 없다
            emit(w, stuckTiles, BitBoard.pieces(new int[][]{{0, 0}}), BitBoard.pieces(new int[][]{{0, 1}, {1, 0}}), 4, new HashMap<>());
            emit(w, stuckTiles, BitBoard.pieces(new int[][]{{0, 0}}), BitBoard.pieces(new int[][]{{0, 1}, {1, 0}}), 6, new HashMap<>());
            // 상대가 둘 차례였다면 막혀 있는 쪽의 말이 상대가 되는 경우 (상대는 움직일 수 있다)
            emit(w, stuckTiles, BitBoard.pieces(new int[][]{{0, 1}, {1, 0}}), BitBoard.pieces(new int[][]{{0, 0}}), 5, new HashMap<>());

            int[][] threat = {
                    {0, 0, 0, 1, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 1, 0, 0, 0},
                    {0, 0, 0, 1, 0, 0, 0},
                    {1, 0, 0, 1, 0, 1, 0}
            };
            long threatTiles = BitBoard.fromArray(threat);
            for (int depth = 2; depth <= 6; depth++) {
                emit(w, threatTiles, BitBoard.pieces(new int[][]{{6, 3}, {6, 5}}), BitBoard.pieces(new int[][]{{0, 3}}), depth, new HashMap<>());
            }
        }
        System.out.println("wrote " + cases + " random cases (" + passes + " with no legal move) + hand-made cases -> " + out);
    }

    private static void emit(PrintWriter w, long tiles, long me, long opp, int depth, Map<Long, Integer> history) {
        tiles |= me | opp;
        int[] buf = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles, me, opp, buf);
        SliderAI ai = new SliderAI(600_000, depth);
        ai.setGameHistory(history);
        SliderAI.Result r = ai.search(tiles, me, opp);
        int[] features = new int[EvalWeights.COUNT];
        boolean immediate = false;
        for (int i = 0; i < n; i++) {
            immediate |= BitBoard.dest(buf[i]) == BitBoard.GOAL;
        }
        if (n > 0 && !immediate) {
            features = new SliderAI(10, 2).features(tiles, me, opp);
        }
        StringBuilder sb = new StringBuilder();
        sb.append(tiles).append(' ').append(me).append(' ').append(opp).append(' ').append(depth);
        sb.append(' ').append(history.size());
        for (Map.Entry<Long, Integer> e : history.entrySet()) {
            sb.append(' ').append(e.getKey()).append(' ').append(e.getValue());
        }
        sb.append(' ').append(n);
        for (int i = 0; i < n; i++) {
            sb.append(' ').append(buf[i]);
        }
        sb.append(' ').append(r.move).append(' ').append(r.score).append(' ').append(r.depth);
        for (int f : features) {
            sb.append(' ').append(f);
        }
        w.println(sb);
    }
}

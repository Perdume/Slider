package org.example.Learning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * AI 끼리 2v2 한 판을 두는 공용 코드. SelfPlay(데이터 생성)와 Match(가중치 비교)가 같이 쓴다.
 * 규칙은 실제 게임과 같다: 한 턴에 말 하나, 어느 말이든 골에 멈추면 승리,
 * 둘 수 있는 수가 없으면 차례 넘김, 같은 국면 3번째 반복은 무승부.
 * 쪽(side) 0 이 먼저 두고 위쪽 귀퉁이에서, 쪽 1 이 아래쪽 귀퉁이에서 시작한다.
 */
final class Arena {
    static final int[][] START_BOARD = {
            {1, 1, 0, 0, 0, 1, 1},
            {1, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 0, 1},
            {1, 1, 0, 0, 0, 1, 1}
    };
    static final int[][] SIDE0_START = {{0, 0}, {0, 6}};
    static final int[][] SIDE1_START = {{6, 0}, {6, 6}};

    static final class Config {
        long timeMs = 30;
        int maxDepth = 40;
        // 이 수가 넘도록 안 끝나면 무승부
        int maxPlies = 150;
        // 처음 몇 수는 무작위로 둬서 시작 국면을 다양하게 한다
        int randomPlies = 4;
        // 그 뒤에도 이 확률로 무작위 수를 둔다 (데이터 다양성용)
        double epsilon = 0.0;
    }

    /** 학습 데이터 한 줄: AI 가 직접 고른 수 앞의 국면. */
    static final class Sample {
        final int game;
        final int ply;
        final int side;
        final int[] features;
        final int score;
        double outcome;

        Sample(int game, int ply, int side, int[] features, int score) {
            this.game = game;
            this.ply = ply;
            this.side = side;
            this.features = features;
            this.score = score;
        }
    }

    static final class Outcome {
        // -1 무승부, 0 먼저 둔 쪽 승, 1 나중에 둔 쪽 승
        final int winner;
        final int plies;
        final List<Sample> samples;

        Outcome(int winner, int plies, List<Sample> samples) {
            this.winner = winner;
            this.plies = plies;
            this.samples = samples;
        }
    }

    private Arena() {
    }

    /**
     * 한 판을 둔다.
     *
     * @param weights     쪽별 평가 가중치 (weights[0] 이 먼저 두는 쪽)
     * @param openingSeed 무작위 시작 수의 시드. 같은 값이면 같은 시작 국면이 된다 (Match 에서 쪽을 바꿔 두 번 둘 때 쓴다)
     * @param noise       epsilon 무작위 수에 쓰는 난수 (null 이면 쓰지 않는다)
     * @param record      true 면 학습 데이터를 모은다
     */
    static Outcome play(Config cfg, EvalWeights[] weights, long openingSeed, Random noise, boolean record, int gameId) {
        Random opening = new Random(openingSeed);
        long tiles = BitBoard.fromArray(START_BOARD);
        long[] pieces = {BitBoard.pieces(SIDE0_START), BitBoard.pieces(SIDE1_START)};
        Map<Long, Integer> counts = new HashMap<>();
        counts.merge(SliderAI.positionHash(tiles, pieces[0], pieces[1]), 1, Integer::sum);
        List<Sample> samples = new ArrayList<>();
        int[] buffer = new int[BitBoard.MAX_MOVES];

        int winner = -1;
        int ply = 0;
        for (; ply < cfg.maxPlies; ply++) {
            int side = ply & 1;
            long me = pieces[side];
            long opp = pieces[1 - side];
            int n = BitBoard.generate(tiles, me, opp, buffer);
            if (n > 0) {
                int move;
                if (ply < cfg.randomPlies) {
                    move = buffer[opening.nextInt(n)];
                } else {
                    SliderAI engine = new SliderAI(cfg.timeMs, cfg.maxDepth, weights[side]);
                    engine.setGameHistory(counts);
                    SliderAI.Result result = engine.search(tiles, me, opp);
                    move = result.move;
                    boolean immediateWin = BitBoard.dest(move) == BitBoard.GOAL;
                    if (record && !immediateWin) {
                        samples.add(new Sample(gameId, ply, side, engine.features(tiles, me, opp), result.score));
                    }
                    if (!immediateWin && noise != null && noise.nextDouble() < cfg.epsilon) {
                        move = buffer[noise.nextInt(n)];
                    }
                }
                tiles = BitBoard.apply(tiles, move);
                pieces[side] = BitBoard.applyPieces(me, move);
                if (BitBoard.dest(move) == BitBoard.GOAL) {
                    winner = side;
                    ply++;
                    break;
                }
            }
            int next = 1 - side;
            int seen = counts.merge(SliderAI.positionHash(tiles, pieces[next], pieces[side]), 1, Integer::sum);
            if (seen >= 3) {
                ply++;
                break;
            }
        }

        for (Sample s : samples) {
            s.outcome = winner < 0 ? 0.5 : winner == s.side ? 1.0 : 0.0;
        }
        return new Outcome(winner, ply, samples);
    }
}

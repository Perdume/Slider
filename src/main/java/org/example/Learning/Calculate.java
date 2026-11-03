package org.example.Learning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Game 과 AI 를 잇는 어댑터. 보드 배열과 말 좌표를 비트보드로 바꿔 탐색하고,
 * 결과를 "어느 말이 어떤 w/a/s/d 를 입력하는지"로 돌려준다.
 */
public class Calculate {
    public static final long DEFAULT_TIME_LIMIT_MS = 1500;
    private static final int MAX_DEPTH = 40;

    private final long tiles;
    private final long aiPieces;
    private final long playerPieces;
    private final long timeLimitMs;
    private Map<Long, Integer> gameHistory;

    private SliderAI.Result result;

    /** playerPieces / aiPieces 는 {row, col} 좌표 목록. */
    public Calculate(int[][] board, int[][] playerPieces, int[][] aiPieces) {
        this(board, playerPieces, aiPieces, DEFAULT_TIME_LIMIT_MS);
    }

    public Calculate(int[][] board, int[][] playerPieces, int[][] aiPieces, long timeLimitMs) {
        this.tiles = BitBoard.fromArray(board);
        this.playerPieces = BitBoard.pieces(playerPieces);
        this.aiPieces = BitBoard.pieces(aiPieces);
        this.timeLimitMs = timeLimitMs;
    }

    /** 말이 한 개씩인 경우. */
    public Calculate(int[][] board, int[] plloc, int[] ailoc) {
        this(board, new int[][]{plloc}, new int[][]{ailoc}, DEFAULT_TIME_LIMIT_MS);
    }

    public Calculate(int[][] board, int[] plloc, int[] ailoc, long timeLimitMs) {
        this(board, new int[][]{plloc}, new int[][]{ailoc}, timeLimitMs);
    }

    /** 실제 게임에서 이미 나온 국면의 횟수 (SliderAI.positionHash 기준, 둘 차례인 쪽 = AI). 3회 반복 무승부 판정에 쓴다. */
    public Calculate withGameHistory(Map<Long, Integer> gameHistory) {
        this.gameHistory = gameHistory;
        return this;
    }

    /**
     * AI 가 이번 턴에 입력할 "w/s/a/d" 목록. 움직일 수 없으면 빈 목록 (차례를 넘긴다).
     * 어느 말이 움직이는지는 getPieceLocation() 으로 알 수 있다.
     */
    public List<String> findsolve() {
        SliderAI ai = new SliderAI(timeLimitMs, MAX_DEPTH, EvalWeights.configured());
        ai.setGameHistory(gameHistory);
        // 시간 제한 대국으로 확인한 개선을 켠다 (Rust 엔진 SPRT: 더 센 LMR 약 Elo +57, 병렬 탐색 4스레드 Elo +100).
        // -Dslider.search=none|lmr|lmr2, -Dslider.threads=N 으로 바꿀 수 있다
        String search = System.getProperty("slider.search", "lmr2");
        ai.setOptions(search.equals("lmr") || search.equals("lmr2"), search.equals("lmr2"));
        ai.setThreads(Integer.getInteger("slider.threads", Runtime.getRuntime().availableProcessors()));
        result = ai.search(tiles, aiPieces, playerPieces);
        if (result.move < 0) {
            return new ArrayList<>();
        }
        return BitBoard.toSymbols(tiles | aiPieces | playerPieces, aiPieces, playerPieces, result.move);
    }

    /**
     * mine 의 각 말이 이번 턴에 둘 수 있는 수(걷기 + 미끄러지기)가 하나라도 있는지.
     * 모두 false 면 그 편은 차례를 넘긴다.
     */
    public static boolean[] movablePieces(int[][] board, int[][] mine, int[][] theirs) {
        long tiles = BitBoard.fromArray(board);
        long me = BitBoard.pieces(mine);
        long opp = BitBoard.pieces(theirs);
        int[] buffer = new int[BitBoard.MAX_MOVES];
        int n = BitBoard.generate(tiles | me | opp, me, opp, buffer);
        boolean[] movable = new boolean[mine.length];
        for (int i = 0; i < n; i++) {
            int src = BitBoard.src(buffer[i]);
            for (int p = 0; p < mine.length; p++) {
                if (BitBoard.cell(mine[p]) == src) {
                    movable[p] = true;
                }
            }
        }
        return movable;
    }

    /** 이번 턴에 움직이는 AI 말의 현재 {row, col}. 차례를 넘기면 null. */
    public int[] getPieceLocation() {
        if (result == null || result.move < 0) {
            return null;
        }
        return BitBoard.loc(BitBoard.src(result.move));
    }

    /** AI 기준 평가 점수 (양수면 AI 유리). */
    public double getScore() {
        return result == null ? 0 : result.score;
    }

    public int getDepth() {
        return result == null ? 0 : result.depth;
    }

    /** 평가 점수를 0~1 사이 값으로 바꾼 대략적인 플레이어 승률. */
    public double getPlayerWinRate() {
        if (result == null) {
            return 0.5;
        }
        if (result.score >= SliderAI.WIN_BOUND) {
            return 0;
        }
        if (result.score <= -SliderAI.WIN_BOUND) {
            return 1;
        }
        return 1 / (1 + Math.exp(result.score / 200.0));
    }
}

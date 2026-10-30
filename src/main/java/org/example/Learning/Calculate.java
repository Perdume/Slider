package org.example.Learning;

import java.util.ArrayList;
import java.util.List;

public class Calculate {
    public static final long DEFAULT_TIME_LIMIT_MS = 1500;
    private static final int MAX_DEPTH = 40;

    private final long tiles;
    private final int aiCell;
    private final int playerCell;
    private final long timeLimitMs;

    private SliderAI.Result result;

    public Calculate(int[][] board, int[] plloc, int[] ailoc) {
        this(board, plloc, ailoc, DEFAULT_TIME_LIMIT_MS);
    }

    public Calculate(int[][] board, int[] plloc, int[] ailoc, long timeLimitMs) {
        tiles = BitBoard.fromArray(board);
        playerCell = BitBoard.cell(plloc);
        aiCell = BitBoard.cell(ailoc);
        this.timeLimitMs = timeLimitMs;
    }

    /** AI(ailoc) 가 이번 턴에 입력할 "w/s/a/d" 목록. 움직일 수 없으면 빈 목록. */
    public List<String> findsolve() {
        result = new SliderAI(timeLimitMs, MAX_DEPTH).search(tiles, aiCell, playerCell);
        if (result.move < 0) {
            return new ArrayList<>();
        }
        return BitBoard.toSymbols(tiles | 1L << aiCell | 1L << playerCell, aiCell, playerCell, result.move);
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

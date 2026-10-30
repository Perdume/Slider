package org.example.Learning;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EvalWeightsTest {

    @Test
    void jsonRoundTripKeepsEveryWeight() {
        EvalWeights original = EvalWeights.of(-151, -399, -49, 2, 3, 140, 66, 31, 13, 2, -4);
        EvalWeights loaded = EvalWeights.fromJson(original.toJson());
        assertArrayEquals(original.toArray(), loaded.toArray());
    }

    @Test
    void missingNamesKeepDefaultsAndNumbersAreRounded() {
        EvalWeights w = EvalWeights.fromJson("{\"mobility\": 4.6, \"tempo\": -2.4}");
        int[] defaults = EvalWeights.defaults().toArray();
        assertEquals(5, w.get(EvalWeights.MOBILITY));
        assertEquals(-2, w.get(EvalWeights.TEMPO));
        assertEquals(defaults[EvalWeights.REACH], w.get(EvalWeights.REACH));
    }

    @Test
    void unknownNameIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> EvalWeights.fromJson("{\"mobilty\": 3}"));
    }

    @Test
    void wrongNumberOfWeightsIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> EvalWeights.of(1, 2, 3));
    }

    @Test
    void namesMatchTheFeatureLayout() {
        assertEquals(EvalWeights.COUNT, EvalWeights.NAMES.length);
        assertEquals("threat1", EvalWeights.NAMES[EvalWeights.THREAT1]);
        assertEquals("plan2", EvalWeights.NAMES[EvalWeights.PLAN_FIRST]);
        assertEquals("position", EvalWeights.NAMES[EvalWeights.POSITION]);
        assertEquals("tempo", EvalWeights.NAMES[EvalWeights.TEMPO]);
    }

    @Test
    void featuresOfSymmetricStartAreBalanced() {
        // 시작 국면은 180도 대칭이라 두 쪽의 특징이 같아서 차이 특징은 모두 0이다
        long tiles = BitBoard.fromArray(Arena.START_BOARD);
        long side0 = BitBoard.pieces(Arena.SIDE0_START);
        long side1 = BitBoard.pieces(Arena.SIDE1_START);
        int[] f = new SliderAI(10, 4).features(tiles, side0, side1);
        assertEquals(EvalWeights.COUNT, f.length);
        assertEquals(0, f[EvalWeights.THREAT1]);
        assertEquals(0, f[EvalWeights.THREAT2]);
        assertEquals(0, f[EvalWeights.MOBILITY]);
        assertEquals(0, f[EvalWeights.REACH]);
        assertEquals(0, f[EvalWeights.POSITION]);
        for (int i = 0; i < 4; i++) {
            assertEquals(0, f[EvalWeights.PLAN_FIRST + i], "plan" + (2 + i));
        }
        assertEquals(1, f[EvalWeights.TEMPO]);
    }

    @Test
    void immediateOpponentThreatShowsUpAsThreatFeature() {
        // 상대(side1) 말 (0,3) 이 아래로 미끄러지면 (4,3) 타일 앞, 즉 골에 멈춘다
        long tiles = BitBoard.fromArray(new int[][]{
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {1, 0, 0, 1, 0, 1, 0}
        });
        long me = BitBoard.pieces(new int[][]{{6, 3}, {6, 5}});
        long opp = BitBoard.pieces(new int[][]{{0, 3}});
        int[] f = new SliderAI(10, 4).features(tiles, me, opp);
        assertEquals(1, f[EvalWeights.THREAT1]);
        assertEquals(0, f[EvalWeights.THREAT2]);
    }

    @Test
    void customWeightsChangeTheSearchScoreButNotLegality() {
        long tiles = BitBoard.fromArray(Arena.START_BOARD);
        long me = BitBoard.pieces(Arena.SIDE1_START);
        long opp = BitBoard.pieces(Arena.SIDE0_START);
        EvalWeights flat = EvalWeights.of(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7);
        SliderAI.Result r = new SliderAI(50, 3, flat).search(tiles, me, opp);
        assertTrue(r.move >= 0);
        // 평가에 쓰이는 특징이 tempo 뿐이라(둘 차례인 쪽 +7) 점수는 +7 또는 -7 이다
        assertEquals(7, Math.abs(r.score));
    }
}

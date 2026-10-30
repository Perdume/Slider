package org.example.Learning;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 평가 함수의 가중치. 평가 점수 = sum(가중치[i] * 특징[i]) 이고, 특징은 항상 "둘 차례인 쪽" 기준이다.
 * 코랩에서 자가대전 데이터로 튜닝한 가중치를 JSON 으로 읽어올 수 있다.
 *
 * JSON 형식 (이름은 NAMES 와 같아야 하고, 빠진 이름은 기본값을 쓴다):
 * {"threat1": -150, "threat2": -400, ...}
 *
 * 특징 정의 (둘 차례인 쪽 = me, 반대편 = opp):
 *  threat1          opp 가 바로 이기는 수가 정확히 1개면 1
 *  threat2          opp 가 바로 이기는 수가 2개 이상이면 1
 *  threat2PerThreat 그 경우의 수의 개수 (2개 이상일 때만, 아니면 0)
 *  mobility         (me 의 이동 가능 수) - (opp 의 이동 가능 수)
 *  reach            (me 말들이 걸을 수 있는 타일 수) - (opp 의 것)
 *  plan2..plan5     골로 들어가는 4방향 직선 중 이기기까지 남은 수(어림)가 2..5인 직선 개수의 me - opp 차이
 *  position         (me 말들의 중앙 근접 점수 합) - (opp 의 것)
 *  tempo            항상 1 (둘 차례인 쪽의 이점)
 */
public final class EvalWeights {
    public static final String[] NAMES = {
            "threat1", "threat2", "threat2PerThreat", "mobility", "reach",
            "plan2", "plan3", "plan4", "plan5", "position", "tempo"
    };
    public static final int COUNT = NAMES.length;

    static final int THREAT1 = 0, THREAT2 = 1, THREAT2_PER = 2, MOBILITY = 3, REACH = 4,
            PLAN_FIRST = 5, POSITION = 9, TEMPO = 10;

    // 튜닝 전에 손으로 정한 값 (이전 버전의 상수와 같다)
    private static final int[] DEFAULTS = {-150, -400, -50, 1, 2, 150, 70, 30, 12, 1, 0};

    private static final Pattern ENTRY = Pattern.compile("\"([A-Za-z0-9_]+)\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)");

    private static EvalWeights configured;

    final int[] values;

    private EvalWeights(int[] values) {
        this.values = values;
    }

    public static EvalWeights defaults() {
        return new EvalWeights(DEFAULTS.clone());
    }

    public static EvalWeights of(int... values) {
        if (values.length != COUNT) {
            throw new IllegalArgumentException("가중치는 " + COUNT + "개여야 합니다: " + values.length);
        }
        return new EvalWeights(values.clone());
    }

    public int get(int index) {
        return values[index];
    }

    public int[] toArray() {
        return values.clone();
    }

    /** 평면 JSON 을 읽는다. 모르는 이름은 오타일 수 있어서 오류로 처리한다. 값은 반올림해서 정수로 쓴다. */
    public static EvalWeights fromJson(String json) {
        int[] values = DEFAULTS.clone();
        Matcher matcher = ENTRY.matcher(json);
        while (matcher.find()) {
            String name = matcher.group(1);
            int index = Arrays.asList(NAMES).indexOf(name);
            if (index < 0) {
                throw new IllegalArgumentException("알 수 없는 가중치 이름: " + name + " (사용 가능: " + Arrays.toString(NAMES) + ")");
            }
            values[index] = (int) Math.round(Double.parseDouble(matcher.group(2)));
        }
        return new EvalWeights(values);
    }

    public static EvalWeights load(Path path) throws IOException {
        return fromJson(new String(Files.readAllBytes(path), StandardCharsets.UTF_8));
    }

    /** "default" 이면 기본값, 아니면 그 경로의 JSON. */
    public static EvalWeights fromArgument(String arg) throws IOException {
        return arg == null || arg.equals("default") ? defaults() : load(Path.of(arg));
    }

    /**
     * 실제 게임이 쓸 가중치. -Dslider.weights=경로 로 JSON 파일을 지정하면 그것을, 없으면 기본값을 쓴다.
     */
    public static synchronized EvalWeights configured() {
        if (configured == null) {
            String path = System.getProperty("slider.weights");
            try {
                configured = path == null || path.isBlank() ? defaults() : load(Path.of(path));
            } catch (IOException e) {
                throw new IllegalStateException("가중치 파일을 읽을 수 없습니다: " + path, e);
            }
        }
        return configured;
    }

    public String toJson() {
        Map<String, Integer> map = new LinkedHashMap<>();
        for (int i = 0; i < COUNT; i++) {
            map.put(NAMES[i], values[i]);
        }
        StringBuilder sb = new StringBuilder("{\n");
        int i = 0;
        for (Map.Entry<String, Integer> e : map.entrySet()) {
            sb.append("  \"").append(e.getKey()).append("\": ").append(e.getValue());
            sb.append(++i < map.size() ? ",\n" : "\n");
        }
        return sb.append("}").toString();
    }

    @Override
    public String toString() {
        return toJson().replace("\n", "");
    }
}

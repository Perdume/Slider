package org.example.Learning;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 자가대전 학습 데이터 생성기 (2v2). 같은 가중치의 AI 끼리 여러 판을 두고,
 * AI 가 직접 고른 수 앞의 국면마다 한 줄(JSONL)씩 기록한다.
 *
 * 한 줄: {"game":0,"ply":12,"side":0,"features":[...],"score":37,"outcome":1.0}
 *  features  EvalWeights.NAMES 순서의 평가 특징 (둘 차례인 쪽 기준)
 *  score     그 국면에서 탐색이 낸 점수 (둘 차례인 쪽 기준, 확정 승패는 +-100000 근처)
 *  outcome   그 판의 최종 결과 (둘 차례였던 쪽 기준: 승 1.0, 무 0.5, 패 0.0)
 *
 * 사용법:
 *   java -cp build org.example.Learning.SelfPlay --games 200 --time-ms 30 --out data/selfplay.jsonl
 * 옵션: --games --time-ms --depth --random-plies --epsilon --max-plies --threads --seed --weights --out
 */
public final class SelfPlay {
    private SelfPlay() {
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = parseArgs(args);
        int games = Integer.parseInt(opt.getOrDefault("games", "200"));
        int threads = Integer.parseInt(opt.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
        long seed = Long.parseLong(opt.getOrDefault("seed", "1"));
        EvalWeights weights = EvalWeights.fromArgument(opt.get("weights"));
        Path out = Path.of(opt.getOrDefault("out", "selfplay.jsonl"));

        Arena.Config cfg = new Arena.Config();
        cfg.timeMs = Long.parseLong(opt.getOrDefault("time-ms", "30"));
        cfg.maxDepth = Integer.parseInt(opt.getOrDefault("depth", "40"));
        cfg.randomPlies = Integer.parseInt(opt.getOrDefault("random-plies", "4"));
        cfg.epsilon = Double.parseDouble(opt.getOrDefault("epsilon", "0.05"));
        cfg.maxPlies = Integer.parseInt(opt.getOrDefault("max-plies", "150"));

        if (out.getParent() != null) {
            Files.createDirectories(out.getParent());
        }
        System.out.printf(Locale.ROOT, "자가대전 시작: games=%d, time=%dms, threads=%d, randomPlies=%d, epsilon=%.2f, weights=%s%n",
                games, cfg.timeMs, threads, cfg.randomPlies, cfg.epsilon, weights);

        AtomicInteger[] tally = {new AtomicInteger(), new AtomicInteger(), new AtomicInteger()}; // 먼저 둔 쪽 승, 무, 나중 쪽 승
        AtomicInteger positions = new AtomicInteger();
        AtomicInteger finished = new AtomicInteger();
        long start = System.currentTimeMillis();

        try (BufferedWriter writer = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
            java.util.List<Future<?>> futures = new java.util.ArrayList<>();
            for (int g = 0; g < games; g++) {
                final int gameId = g;
                futures.add(pool.submit(() -> {
                    long gameSeed = seed * 1_000_003L + gameId;
                    Arena.Outcome outcome = Arena.play(cfg, new EvalWeights[]{weights, weights}, gameSeed,
                            new Random(gameSeed ^ 0x5DEECE66DL), true, gameId);
                    tally[outcome.winner < 0 ? 1 : outcome.winner == 0 ? 0 : 2].incrementAndGet();
                    positions.addAndGet(outcome.samples.size());
                    StringBuilder sb = new StringBuilder();
                    for (Arena.Sample s : outcome.samples) {
                        sb.append(toJson(s)).append('\n');
                    }
                    synchronized (writer) {
                        try {
                            writer.write(sb.toString());
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    int done = finished.incrementAndGet();
                    if (done % 10 == 0 || done == games) {
                        System.out.printf(Locale.ROOT, "  %d/%d판 (국면 %d개, %.0f초)%n", done, games, positions.get(),
                                (System.currentTimeMillis() - start) / 1000.0);
                    }
                }));
            }
            pool.shutdown();
            for (Future<?> f : futures) {
                f.get();
            }
        }
        System.out.printf(Locale.ROOT, "끝: 먼저 둔 쪽 %d승 / 무승부 %d / 나중 쪽 %d승, 국면 %d개 -> %s%n",
                tally[0].get(), tally[1].get(), tally[2].get(), positions.get(), out);
    }

    private static String toJson(Arena.Sample s) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("{\"game\":").append(s.game).append(",\"ply\":").append(s.ply).append(",\"side\":").append(s.side)
                .append(",\"features\":[");
        for (int i = 0; i < s.features.length; i++) {
            sb.append(i == 0 ? "" : ",").append(s.features[i]);
        }
        return sb.append("],\"score\":").append(s.score).append(",\"outcome\":").append(s.outcome).append('}').toString();
    }

    static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException("옵션은 '--이름 값' 형식이어야 합니다: " + args[i]);
            }
            map.put(args[i].substring(2), args[++i]);
        }
        return map;
    }
}

package org.example.Learning;

import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 가중치 두 벌(A, B)을 맞붙이는 벤치마크 (2v2). 시작 국면 하나마다 쪽을 바꿔 두 판씩 둬서 선공/후공 차이를 없앤다.
 *
 * 사용법:
 *   java -cp build org.example.Learning.Match --a weights.json --b default --games 100 --time-ms 50
 * 옵션: --a --b (JSON 경로 또는 default) --games (짝수로 올림) --time-ms --depth --random-plies --max-plies --threads --seed
 */
public final class Match {
    private Match() {
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = SelfPlay.parseArgs(args);
        EvalWeights a = EvalWeights.fromArgument(opt.get("a"));
        EvalWeights b = EvalWeights.fromArgument(opt.get("b"));
        int pairs = (Integer.parseInt(opt.getOrDefault("games", "100")) + 1) / 2;
        int threads = Integer.parseInt(opt.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
        long seed = Long.parseLong(opt.getOrDefault("seed", "1"));

        Arena.Config cfg = new Arena.Config();
        cfg.timeMs = Long.parseLong(opt.getOrDefault("time-ms", "50"));
        cfg.maxDepth = Integer.parseInt(opt.getOrDefault("depth", "40"));
        // 시작 국면을 다양하게 하려고 처음 몇 수는 무작위로 둔다 (한 쌍의 두 판은 같은 시작 국면)
        cfg.randomPlies = Integer.parseInt(opt.getOrDefault("random-plies", "6"));
        cfg.maxPlies = Integer.parseInt(opt.getOrDefault("max-plies", "150"));
        cfg.epsilon = 0;

        System.out.printf("A = %s%nB = %s%n%d판 (시작 국면 %d개 x 쪽 교대), time=%dms, threads=%d%n",
                a, b, pairs * 2, pairs, cfg.timeMs, threads);

        AtomicInteger aWins = new AtomicInteger();
        AtomicInteger bWins = new AtomicInteger();
        AtomicInteger draws = new AtomicInteger();
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
        java.util.List<Future<?>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < pairs; i++) {
            final int pair = i;
            futures.add(pool.submit(() -> {
                long openingSeed = seed * 7919L + pair;
                // 첫 판: A 가 먼저 두는 쪽, 둘째 판: B 가 먼저 두는 쪽
                Arena.Outcome first = Arena.play(cfg, new EvalWeights[]{a, b}, openingSeed, null, false, pair * 2);
                Arena.Outcome second = Arena.play(cfg, new EvalWeights[]{b, a}, openingSeed, null, false, pair * 2 + 1);
                count(first.winner, true, aWins, bWins, draws);
                count(second.winner, false, aWins, bWins, draws);
            }));
        }
        pool.shutdown();
        for (Future<?> f : futures) {
            f.get();
        }

        int games = pairs * 2;
        double score = (aWins.get() + 0.5 * draws.get()) / games;
        // 판마다 결과가 1/0.5/0 이므로 표본 분산으로 대략적인 95% 구간을 낸다
        double var = (aWins.get() * sq(1 - score) + draws.get() * sq(0.5 - score) + bWins.get() * sq(score)) / games;
        double margin = 1.96 * Math.sqrt(var / games);
        System.out.printf(Locale.ROOT, "A %d승 / 무 %d / B %d승  ->  A 점수 %.3f (95%% 구간 약 %.3f ~ %.3f)%n",
                aWins.get(), draws.get(), bWins.get(), score, score - margin, score + margin);
        // 노트북 등 다른 프로그램이 읽기 쉬운 한 줄
        System.out.printf(Locale.ROOT, "RESULT {\"aWins\":%d,\"draws\":%d,\"bWins\":%d,\"score\":%.4f,\"low\":%.4f,\"high\":%.4f}%n",
                aWins.get(), draws.get(), bWins.get(), score, score - margin, score + margin);
    }

    // 먼저 둔 쪽이 이기면 winner 0. aFirst 면 먼저 둔 쪽이 A.
    private static void count(int winner, boolean aFirst, AtomicInteger aWins, AtomicInteger bWins, AtomicInteger draws) {
        if (winner < 0) {
            draws.incrementAndGet();
        } else if ((winner == 0) == aFirst) {
            aWins.incrementAndGet();
        } else {
            bWins.incrementAndGet();
        }
    }

    private static double sq(double x) {
        return x * x;
    }
}

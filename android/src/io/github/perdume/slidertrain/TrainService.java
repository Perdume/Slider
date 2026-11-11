package io.github.perdume.slidertrain;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.IBinder;
import android.os.PowerManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 학습을 백그라운드에서 돌리는 포그라운드 서비스.
 * 실제 일은 Rust 실행 파일 libspsa.so 가 한다 (대국을 반복하며 탐색 상수 + 평가 가중치를 SPSA 로 맞춤, 새 규칙).
 * 이 서비스는 그 프로세스를 띄우고, 출력 줄을 화면에 넘기고, 중지 신호(STOP 파일)를 만든다.
 */
public class TrainService extends Service {
    static final String ACTION_START = "io.github.perdume.slidertrain.START";
    static final String ACTION_STOP = "io.github.perdume.slidertrain.STOP";
    private static final String CHANNEL = "train";
    private static final int NOTIFICATION_ID = 1;
    private static final int MAX_LOG_CHARS = 200_000;

    /** 화면이 읽는 상태. 서비스가 죽었다 살아나도 화면에 마지막 로그가 남도록 정적으로 둔다. */
    static final StringBuilder LOG = new StringBuilder();
    static volatile boolean running = false;
    static volatile Runnable onChange;

    private volatile Process process;
    private PowerManager.WakeLock wakeLock;
    private File workDir;

    /** SPSA 진행 상태와 결과가 쌓이는 폴더 (다시 시작하면 여기서 이어서) */
    static File outDir(Context c) {
        File dir = new File(workDir(c), "spsa");
        dir.mkdirs();
        return dir;
    }

    static File workDir(Context c) {
        File dir = new File(c.getFilesDir(), "work");
        dir.mkdirs();
        return dir;
    }

    static void log(String line) {
        synchronized (LOG) {
            LOG.append(line).append('\n');
            if (LOG.length() > MAX_LOG_CHARS) {
                LOG.delete(0, LOG.length() - MAX_LOG_CHARS / 2);
            }
        }
        notifyChange();
    }

    static void notifyChange() {
        Runnable r = onChange;
        if (r != null) {
            r.run();
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? null : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            requestStop();
            return START_NOT_STICKY;
        }
        if (!ACTION_START.equals(action) || running) {
            return START_NOT_STICKY;
        }
        running = true;
        workDir = workDir(this);
        startForeground(NOTIFICATION_ID, notification("watch".equals(intent.getStringExtra("mode")) ? "AI 대국을 둡니다" : "튜닝을 시작합니다"));
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "slidertrain:train");
        wakeLock.acquire(24L * 60 * 60 * 1000);
        final Intent request = intent;
        new Thread(() -> runTraining(request), "train-runner").start();
        notifyChange();
        return START_NOT_STICKY;
    }

    private void runTraining(Intent request) {
        try {
            File stopFile = new File(workDir, "STOP");
            stopFile.delete();
            List<String> cmd = buildCommand(request, stopFile);
            log("$ " + String.join(" ", cmd));
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.directory(workDir);
            pb.redirectErrorStream(true);
            process = pb.start();
            long lastNote = 0;
            try (BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = in.readLine()) != null) {
                    log(line);
                    long now = System.currentTimeMillis();
                    if (now - lastNote > 5000 && !line.trim().isEmpty() && !line.startsWith("$")) {
                        lastNote = now;
                        updateNotification(line.trim());
                    }
                }
            }
            int code = process.waitFor();
            log(code == 0 ? "== 끝났습니다 ==" : "== 비정상 종료 (코드 " + code + ") ==");
        } catch (Exception e) {
            log("오류: " + e);
        } finally {
            running = false;
            process = null;
            if (wakeLock != null && wakeLock.isHeld()) {
                wakeLock.release();
            }
            stopForeground(true);
            stopSelf();
            notifyChange();
        }
    }

    private List<String> buildCommand(Intent r, File stopFile) throws IOException {
        File lib = new File(getApplicationInfo().nativeLibraryDir);
        if ("watch".equals(r.getStringExtra("mode"))) {
            return watchCommand(r, lib, stopFile);
        }
        File bin = new File(lib, "libspsa.so");
        if (!bin.exists()) {
            throw new IOException("실행 파일이 없습니다 (이 기기는 arm64 가 아닐 수 있습니다): " + bin);
        }
        File out = outDir(this);
        if (r.getBooleanExtra("reset", false)) {
            new File(out, "spsa_state.txt").delete();
            log("== 저장된 진행을 지우고 처음부터 ==");
        }
        List<String> c = new ArrayList<>();
        c.add(bin.getAbsolutePath());
        add(c, "--out", out.getAbsolutePath());
        add(c, "--stop-file", stopFile.getAbsolutePath());
        add(c, "--games", String.valueOf(r.getIntExtra("games", 32)));
        add(c, "--time-ms", String.valueOf(r.getIntExtra("timeMs", 30)));
        add(c, "--threads", String.valueOf(r.getIntExtra("threads", 2)));
        add(c, "--iters", String.valueOf(r.getIntExtra("iters", 100000)));
        String tune = r.getStringExtra("tune");
        if (tune != null && !tune.trim().isEmpty()) {
            add(c, "--tune", tune.trim().replaceAll("\\s+", ""));
        }
        String weights = r.getStringExtra("weightsJson");
        if (weights != null && !weights.trim().isEmpty()) {
            File w = new File(workDir, "weights.json");
            try (FileOutputStream fo = new FileOutputStream(w)) {
                fo.write(weights.getBytes(StandardCharsets.UTF_8));
            }
            add(c, "--weights", w.getAbsolutePath());
        }
        return c;
    }

    /** AI 끼리 한 판 (libwatch.so): 매 수마다 판을 로그로 */
    private List<String> watchCommand(Intent r, File lib, File stopFile) throws IOException {
        File bin = new File(lib, "libwatch.so");
        if (!bin.exists()) {
            throw new IOException("실행 파일이 없습니다: " + bin);
        }
        List<String> c = new ArrayList<>();
        c.add(bin.getAbsolutePath());
        add(c, "--time-ms", String.valueOf(r.getIntExtra("watchMs", 500)));
        add(c, "--delay-ms", "700");
        add(c, "--stop-file", stopFile.getAbsolutePath());
        add(c, "--threads", String.valueOf(r.getIntExtra("threads", 2)));
        add(c, "--random-plies", "2");
        add(c, "--seed", String.valueOf(System.currentTimeMillis() % 100000));
        if (r.getBooleanExtra("watchTuned", false)) {
            File out = outDir(this);
            File t = new File(out, "spsa_tune.txt");
            File w = new File(out, "spsa_weights.json");
            if (t.exists() && w.exists()) {
                byte[] b = java.nio.file.Files.readAllBytes(t.toPath());
                add(c, "--tune-a", new String(b, StandardCharsets.UTF_8).trim());
                add(c, "--weights-a", w.getAbsolutePath());
            } else {
                log("(튜닝 결과가 아직 없어서 양쪽 모두 게임 기본값으로 둡니다)");
            }
        }
        return c;
    }

    private static void add(List<String> c, String key, String value) {
        c.add(key);
        c.add(value);
    }

    /** STOP 파일을 만들어 깨끗하게 멈추게 한다. 30초 안에 안 끝나면 강제 종료한다. */
    private void requestStop() {
        if (!running) {
            return;
        }
        try {
            new File(workDir(this), "STOP").createNewFile();
        } catch (IOException ignored) {
        }
        log("== 중지 요청: 진행 중인 한 판이 끝나는 대로 멈춥니다 ==");
        final Process p = process;
        new Thread(() -> {
            try {
                Thread.sleep(30_000);
            } catch (InterruptedException ignored) {
            }
            if (p != null && p.isAlive()) {
                p.destroyForcibly();
                log("== 30초 안에 멈추지 않아 강제 종료했습니다 ==");
            }
        }, "stop-watchdog").start();
    }

    private Notification notification(String text) {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "학습 진행", NotificationManager.IMPORTANCE_LOW));
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this, CHANNEL)
                .setContentTitle("Slider 튜닝 중")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.stat_sys_download)
                .setContentIntent(open)
                .setOngoing(true)
                .build();
    }

    private void updateNotification(String text) {
        NotificationManager nm = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        nm.notify(NOTIFICATION_ID, notification(text));
    }

    @Override
    public void onDestroy() {
        Process p = process;
        if (p != null && p.isAlive()) {
            try {
                new File(workDir(this), "STOP").createNewFile();
            } catch (IOException ignored) {
            }
            p.destroyForcibly();
        }
        super.onDestroy();
    }
}

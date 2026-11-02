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
 * 실제 일은 Rust 실행 파일 libtrainloop.so 가 한다 (자가대전 -> 신경망 학습 -> 대국 확인 -> 채택).
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
        startForeground(NOTIFICATION_ID, notification("학습을 시작합니다"));
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "slidertrain:train");
        wakeLock.acquire(12L * 60 * 60 * 1000);
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
        File loop = new File(lib, "libtrainloop.so");
        if (!loop.exists()) {
            throw new IOException("실행 파일이 없습니다 (이 기기는 arm64 가 아닐 수 있습니다): " + loop);
        }
        List<String> c = new ArrayList<>();
        c.add(loop.getAbsolutePath());
        add(c, "--work-dir", workDir.getAbsolutePath());
        add(c, "--selfplay-bin", new File(lib, "libselfplay.so").getAbsolutePath());
        add(c, "--trainnet-bin", new File(lib, "libtrainnet.so").getAbsolutePath());
        add(c, "--matchup-bin", new File(lib, "libmatchup.so").getAbsolutePath());
        add(c, "--stop-file", stopFile.getAbsolutePath());
        add(c, "--rounds", String.valueOf(r.getIntExtra("rounds", 2)));
        add(c, "--games", String.valueOf(r.getIntExtra("games", 600)));
        add(c, "--time-ms", String.valueOf(r.getIntExtra("timeMs", 30)));
        add(c, "--threads", String.valueOf(r.getIntExtra("threads", 2)));
        add(c, "--epochs", String.valueOf(r.getIntExtra("epochs", 40)));
        add(c, "--match-games", String.valueOf(r.getIntExtra("matchGames", 200)));
        add(c, "--match-time-ms", String.valueOf(r.getIntExtra("matchTimeMs", 50)));
        add(c, "--lambda", String.valueOf(r.getFloatExtra("lambda", 1.0f)));
        String weights = r.getStringExtra("weightsJson");
        if (weights != null && !weights.trim().isEmpty()) {
            File w = new File(workDir, "weights.json");
            try (FileOutputStream out = new FileOutputStream(w)) {
                out.write(weights.getBytes(StandardCharsets.UTF_8));
            }
            add(c, "--weights", w.getAbsolutePath());
        }
        File prev = new File(workDir, "net.txt");
        if (r.getBooleanExtra("resume", false) && prev.exists()) {
            add(c, "--net", prev.getAbsolutePath());
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
                .setContentTitle("Slider 학습 중")
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

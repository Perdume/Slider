package io.github.perdume.slidertrain;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** 설정을 입력하고 학습을 시작/중지하는 화면. 화면은 코드로 만든다 (리소스 파일 없음). */
public class MainActivity extends Activity {
    private SharedPreferences prefs;
    private EditText rounds, games, timeMs, threads, epochs, matchGames, matchTimeMs, lambda, weights;
    private CheckBox resume;
    private TextView status, logView;
    private ScrollView logScroll;
    private Button start, stop;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("settings", MODE_PRIVATE);
        int cores = Runtime.getRuntime().availableProcessors();

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(12);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("Slider 신경망 학습");
        title.setTextSize(22);
        root.addView(title);

        TextView note = new TextView(this);
        note.setText("이 기기 CPU 코어 " + cores + "개로 자가대전을 돌려 신경망을 학습합니다. NPU/GPU 는 쓰지 않습니다 (탐색이 정수 분기 위주라 가속 이득이 없습니다).");
        note.setTextSize(13);
        root.addView(note);

        rounds = field(root, "라운드 수", "rounds", "2");
        games = field(root, "라운드당 자가대전 판 수", "games", "600");
        timeMs = field(root, "수당 탐색 시간(ms)", "timeMs", "30");
        threads = field(root, "스레드 수 (발열이 심하면 줄이세요)", "threads", String.valueOf(Math.max(1, cores - 1)));
        epochs = field(root, "학습 epoch", "epochs", "40");
        matchGames = field(root, "대국 확인 판 수", "matchGames", "200");
        matchTimeMs = field(root, "대국 확인 수당 시간(ms)", "matchTimeMs", "50");
        lambda = field(root, "탐색 점수 비중 λ (0~1)", "lambda", "1.0");
        lambda.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);

        TextView wl = new TextView(this);
        wl.setText("선형 가중치 JSON (선택, 비우면 기본값)");
        root.addView(wl);
        weights = new EditText(this);
        weights.setMinLines(2);
        weights.setMaxLines(6);
        weights.setHorizontallyScrolling(false);
        weights.setText(prefs.getString("weights", ""));
        weights.setHint("{\"threat1\": -620, ...}");
        root.addView(weights);

        resume = new CheckBox(this);
        resume.setText("이전에 저장된 net.txt 에서 이어서");
        resume.setChecked(prefs.getBoolean("resume", false));
        root.addView(resume);

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);
        start = button("시작", v -> startTraining());
        stop = button("중지", v -> startService(new Intent(this, TrainService.class).setAction(TrainService.ACTION_STOP)));
        buttons.addView(start, weight());
        buttons.addView(stop, weight());
        root.addView(buttons);

        LinearLayout buttons2 = new LinearLayout(this);
        buttons2.setOrientation(LinearLayout.HORIZONTAL);
        buttons2.addView(button("결과를 다운로드 폴더에 저장", v -> saveResult()), weight());
        buttons2.addView(button("로그 지우기", v -> {
            synchronized (TrainService.LOG) {
                TrainService.LOG.setLength(0);
            }
            refresh();
        }), weight());
        root.addView(buttons2);

        status = new TextView(this);
        status.setTextSize(15);
        status.setGravity(Gravity.START);
        root.addView(status);

        logView = new TextView(this);
        logView.setTypeface(android.graphics.Typeface.MONOSPACE);
        logView.setTextSize(11);
        logView.setTextIsSelectable(true);
        root.addView(logView);

        ScrollView settingsScroll = new ScrollView(this);
        logScroll = settingsScroll;
        settingsScroll.addView(root, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(settingsScroll);
    }

    @Override
    protected void onResume() {
        super.onResume();
        TrainService.onChange = () -> runOnUiThread(this::refresh);
        refresh();
    }

    @Override
    protected void onPause() {
        TrainService.onChange = null;
        saveSettings();
        super.onPause();
    }

    private void startTraining() {
        if (TrainService.running) {
            return;
        }
        saveSettings();
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            // 알림을 허용하지 않아도 학습은 돌지만, 진행 상황 알림이 보이지 않는다
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1);
        }
        Intent i = new Intent(this, TrainService.class).setAction(TrainService.ACTION_START);
        i.putExtra("rounds", intOf(rounds, 2));
        i.putExtra("games", intOf(games, 600));
        i.putExtra("timeMs", intOf(timeMs, 30));
        i.putExtra("threads", Math.max(1, intOf(threads, 2)));
        i.putExtra("epochs", intOf(epochs, 40));
        i.putExtra("matchGames", intOf(matchGames, 200));
        i.putExtra("matchTimeMs", intOf(matchTimeMs, 50));
        try {
            i.putExtra("lambda", Float.parseFloat(lambda.getText().toString().trim()));
        } catch (NumberFormatException e) {
            i.putExtra("lambda", 1.0f);
        }
        i.putExtra("weightsJson", weights.getText().toString());
        i.putExtra("resume", resume.isChecked());
        startForegroundService(i);
    }

    private void refresh() {
        boolean running = TrainService.running;
        start.setEnabled(!running);
        stop.setEnabled(running);
        File net = new File(TrainService.workDir(this), "net.txt");
        status.setText((running ? "● 실행 중" : "○ 대기") + (net.exists() ? "   |   저장된 net.txt " + net.length() / 1024 + " KB" : ""));
        String text;
        synchronized (TrainService.LOG) {
            text = TrainService.LOG.toString();
        }
        logView.setText(text);
        logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
    }

    /** 결과 파일(net.txt)과 로그를 다운로드 폴더에 복사한다 (저장소 권한 없이 MediaStore 로). */
    private void saveResult() {
        File work = TrainService.workDir(this);
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        int saved = 0;
        try {
            File net = new File(work, "net.txt");
            if (net.exists()) {
                copyToDownloads(net, "slider_net_" + stamp + ".txt");
                saved++;
            }
            File summary = new File(work, "summary.txt");
            if (summary.exists()) {
                copyToDownloads(summary, "slider_summary_" + stamp + ".txt");
                saved++;
            }
            String log;
            synchronized (TrainService.LOG) {
                log = TrainService.LOG.toString();
            }
            if (!log.isEmpty()) {
                File tmp = new File(getCacheDir(), "log.txt");
                try (java.io.FileOutputStream out = new java.io.FileOutputStream(tmp)) {
                    out.write(log.getBytes("UTF-8"));
                }
                copyToDownloads(tmp, "slider_log_" + stamp + ".txt");
                saved++;
            }
        } catch (IOException e) {
            Toast.makeText(this, "저장 실패: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return;
        }
        Toast.makeText(this, saved == 0 ? "저장할 결과가 아직 없습니다" : "다운로드 폴더에 " + saved + "개 저장했습니다", Toast.LENGTH_LONG).show();
    }

    private void copyToDownloads(File src, String name) throws IOException {
        ContentValues v = new ContentValues();
        v.put(MediaStore.Downloads.DISPLAY_NAME, name);
        v.put(MediaStore.Downloads.MIME_TYPE, "text/plain");
        v.put(MediaStore.Downloads.RELATIVE_PATH, "Download/");
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
        if (uri == null) {
            throw new IOException("다운로드 폴더에 파일을 만들 수 없습니다");
        }
        try (InputStream in = new FileInputStream(src); OutputStream out = getContentResolver().openOutputStream(uri)) {
            byte[] buf = new byte[16 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
        }
    }

    private void saveSettings() {
        prefs.edit()
                .putString("rounds", rounds.getText().toString())
                .putString("games", games.getText().toString())
                .putString("timeMs", timeMs.getText().toString())
                .putString("threads", threads.getText().toString())
                .putString("epochs", epochs.getText().toString())
                .putString("matchGames", matchGames.getText().toString())
                .putString("matchTimeMs", matchTimeMs.getText().toString())
                .putString("lambda", lambda.getText().toString())
                .putString("weights", weights.getText().toString())
                .putBoolean("resume", resume.isChecked())
                .apply();
    }

    private EditText field(LinearLayout parent, String label, String key, String def) {
        TextView t = new TextView(this);
        t.setText(label);
        parent.addView(t);
        EditText e = new EditText(this);
        e.setInputType(InputType.TYPE_CLASS_NUMBER);
        e.setSingleLine(true);
        e.setText(prefs.getString(key, def));
        parent.addView(e);
        return e;
    }

    private Button button(String text, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setOnClickListener(l);
        return b;
    }

    private LinearLayout.LayoutParams weight() {
        return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
    }

    private int intOf(EditText e, int def) {
        try {
            return Integer.parseInt(e.getText().toString().trim());
        } catch (NumberFormatException ex) {
            return def;
        }
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }
}

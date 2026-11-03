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

/** 설정을 입력하고 튜닝(SPSA)을 시작/중지하는 화면. 화면은 코드로 만든다 (리소스 파일 없음). */
public class MainActivity extends Activity {
    private SharedPreferences prefs;
    private EditText games, timeMs, threads, iters, tune, weights, watchMs;
    private CheckBox reset, watchTuned;
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
        title.setText("Slider AI 튜닝 (새 규칙)");
        title.setTextSize(22);
        root.addView(title);

        TextView note = new TextView(this);
        note.setText("이 기기 CPU 코어 " + cores + "개로 대국을 반복하며 AI 의 탐색 상수와 평가 가중치를 조금씩 맞춥니다 (SPSA). "
                + "멈췄다가 다시 시작하면 이어서 합니다. 충전기를 꽂고 돌리세요. 결과(spsa_tune / spsa_weights)는 다운로드 폴더에 저장할 수 있습니다.");
        note.setTextSize(13);
        root.addView(note);

        games = field(root, "반복당 대국 수 (짝수, 많을수록 정확하고 느림)", "games", "32");
        timeMs = field(root, "수당 탐색 시간(ms)", "timeMs", "30");
        threads = field(root, "스레드 수 (발열이 심하면 줄이세요)", "threads", String.valueOf(Math.max(1, cores - 1)));
        iters = field(root, "최대 반복 수", "iters", "100000");

        TextView tl = new TextView(this);
        tl.setText("시작 탐색 상수 (선택, 비우면 게임 설정. 예: margin=77,trap=24,...)");
        root.addView(tl);
        tune = new EditText(this);
        tune.setSingleLine(false);
        tune.setText(prefs.getString("tune", ""));
        root.addView(tune);

        TextView wl = new TextView(this);
        wl.setText("시작 가중치 JSON (선택, 비우면 게임 설정)");
        root.addView(wl);
        weights = new EditText(this);
        weights.setMinLines(2);
        weights.setMaxLines(6);
        weights.setHorizontallyScrolling(false);
        weights.setText(prefs.getString("weights", ""));
        weights.setHint("{\"threat1\": -620, ...}");
        root.addView(weights);

        reset = new CheckBox(this);
        reset.setText("처음부터 다시 (저장된 진행을 지우고 위 시작값에서)");
        reset.setChecked(false);
        root.addView(reset);

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);
        start = button("시작", v -> startTraining());
        stop = button("중지", v -> startService(new Intent(this, TrainService.class).setAction(TrainService.ACTION_STOP)));
        buttons.addView(start, weight());
        buttons.addView(stop, weight());
        root.addView(buttons);

        TextView wt = new TextView(this);
        wt.setText("\nAI 대국 보기: AI 끼리 한 판을 두며 매 수마다 판, 움직인 말, 형세 점수를 아래 로그에 보여 줍니다.");
        root.addView(wt);
        watchMs = field(root, "대국 보기 수당 생각 시간(ms)", "watchMs", "500");
        watchTuned = new CheckBox(this);
        watchTuned.setText("A 쪽(말 1, 2)은 튜닝 중인 값으로 (B 쪽 X, Y 는 게임 기본값)");
        watchTuned.setChecked(prefs.getBoolean("watchTuned", false));
        root.addView(watchTuned);
        root.addView(button("AI 대국 보기", v -> startWatch()));

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
        i.putExtra("games", Math.max(2, intOf(games, 32)));
        i.putExtra("timeMs", intOf(timeMs, 30));
        i.putExtra("threads", Math.max(1, intOf(threads, 2)));
        i.putExtra("iters", intOf(iters, 100000));
        i.putExtra("tune", tune.getText().toString().trim());
        i.putExtra("weightsJson", weights.getText().toString());
        i.putExtra("reset", reset.isChecked());
        reset.setChecked(false);
        startForegroundService(i);
    }

    private void startWatch() {
        if (TrainService.running) {
            Toast.makeText(this, "튜닝이 돌고 있습니다. 중지한 뒤에 보세요", Toast.LENGTH_LONG).show();
            return;
        }
        saveSettings();
        synchronized (TrainService.LOG) {
            TrainService.LOG.setLength(0);
        }
        Intent i = new Intent(this, TrainService.class).setAction(TrainService.ACTION_START);
        i.putExtra("mode", "watch");
        i.putExtra("watchMs", Math.max(10, intOf(watchMs, 500)));
        i.putExtra("threads", Math.max(1, intOf(threads, 2)));
        i.putExtra("watchTuned", watchTuned.isChecked());
        startForegroundService(i);
    }

    private void refresh() {
        boolean running = TrainService.running;
        start.setEnabled(!running);
        stop.setEnabled(running);
        File state = new File(TrainService.outDir(this), "spsa_state.txt");
        String done = "";
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(state))) {
            done = "   |   지금까지 반복 " + r.readLine().trim();
        } catch (Exception ignored) {
        }
        status.setText((running ? "● 실행 중" : "○ 대기") + done);
        String text;
        synchronized (TrainService.LOG) {
            text = TrainService.LOG.toString();
        }
        logView.setText(text);
        logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
    }

    /** 결과 파일(net.txt)과 로그를 다운로드 폴더에 복사한다 (저장소 권한 없이 MediaStore 로). */
    private void saveResult() {
        File work = TrainService.outDir(this);
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        int saved = 0;
        try {
            for (String[] f : new String[][]{{"spsa_tune.txt", "slider_spsa_tune_"}, {"spsa_weights.json", "slider_spsa_weights_"}}) {
                File src = new File(work, f[0]);
                if (src.exists()) {
                    copyToDownloads(src, f[1] + stamp + (f[0].endsWith(".json") ? ".json" : ".txt"));
                    saved++;
                }
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
                .putString("games", games.getText().toString())
                .putString("timeMs", timeMs.getText().toString())
                .putString("threads", threads.getText().toString())
                .putString("iters", iters.getText().toString())
                .putString("tune", tune.getText().toString())
                .putString("weights", weights.getText().toString())
                .putString("watchMs", watchMs.getText().toString())
                .putBoolean("watchTuned", watchTuned.isChecked())
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

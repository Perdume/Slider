"""Texel 튜닝 (colab/slider_2v2_training.ipynb 의 학습 코드와 같음). all_experiments.py 가 가져다 쓴다."""
import json

import numpy as np
import pandas as pd

from scipy.optimize import minimize

# EvalWeights.java 의 NAMES / 기본값과 같아야 합니다 (데이터를 읽을 때 길이를 확인합니다)
NAMES = ["threat1", "threat2", "threat2PerThreat", "mobility", "reach",
         "plan2", "plan3", "plan4", "plan5", "position", "tempo"]
DEFAULT_WEIGHTS = dict(zip(NAMES, [-150, -400, -50, 1, 2, 150, 70, 30, 12, 1, 0]))
FORCED_SCORE = 50_000  # 이 이상이면 탐색이 승패를 확정한 국면
LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
REG_UNIT = 200.0       # 정규화에서 가중치 차이를 재는 단위
MAX_ABS_WEIGHT = 3000


def write_weights(path, w):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: int(w[k]) for k in NAMES}, f, indent=2)


def to_vector(w):
    return np.array([w[k] for k in NAMES], dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def mse(w, X, y, scale):
    return float(np.mean((sigmoid(X @ w / scale) - y) ** 2))


K_MIN, K_MAX = 20.0, 20000.0


def calibrate_scale(w, X, y):
    """가중치 w 를 고정하고, P(승) = sigmoid(점수 / K) 의 K 를 오차가 가장 작은 값으로 고른다."""
    grid = np.exp(np.linspace(np.log(K_MIN), np.log(K_MAX), 300))
    errs = [mse(w, X, y, k) for k in grid]
    return float(grid[int(np.argmin(errs))])


def load_samples(files):
    """files = [(라운드, jsonl 경로), ...]  ->  X(특징), y(결과), gid(판 번호; 라운드가 달라도 겹치지 않음)"""
    frames = []
    for rnd, path in files:
        df = pd.read_json(path, lines=True)
        df["gid"] = rnd * 1_000_000 + df["game"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    X = np.array(df["features"].tolist(), dtype=float)
    if X.shape[1] != len(NAMES):
        raise ValueError(f"특징이 {X.shape[1]}개인데 NAMES 는 {len(NAMES)}개입니다. EvalWeights.java 와 맞춰 주세요.")
    keep = (df["score"].abs() < FORCED_SCORE).values
    print(f"국면 {len(df):,}개 중 확정 승패 {(~keep).sum():,}개 제외 -> {keep.sum():,}개 사용 "
          f"(판 {df['gid'].nunique()}개, 승 {np.mean(df['outcome'] == 1):.1%} / 무 {np.mean(df['outcome'] == 0.5):.1%} / 패 {np.mean(df['outcome'] == 0):.1%})")
    return (X[keep], df["outcome"].values[keep].astype(float), df["gid"].values[keep],
            df["score"].values[keep].astype(float))


def corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def diagnose(X, y, t, w0):
    """데이터에 학습할 만한 신호가 있는지 보여 준다. 상관계수가 0 에 가까우면 가중치를 바꿔도 얻을 게 적다."""
    ev = X @ w0
    print(f"신호 세기 (상관계수): 평가점수-승패 {corr(ev, y):+.3f} | 탐색점수-승패 {corr(t, y):+.3f} | 평가점수-탐색점수 {corr(ev, t):+.3f}")
    print(f"점수 표준편차: 평가점수 {ev.std():.0f} / 탐색점수 {t.std():.0f}  (평가가 훨씬 크면 평가가 탐색보다 과하게 확신한다는 뜻)")


def fit(X, y, w0, lam, scale):
    """제곱오차 + lam * |(w - w0) / REG_UNIT|^2 를 최소화하는 가중치. 눈금 scale 은 고정."""
    s = X.std(axis=0)
    s[s < 1e-9] = 1.0              # 상수 열 (tempo 등) 은 그대로 둔다
    Xs = X / s                     # 열 크기를 맞춘다

    # 변수 v = w * s / scale 로 바꿔서 기울기가 O(0.01~1) 이 되게 한다.
    # (w 그대로 쓰면 기울기가 1e-5 쯤이라 L-BFGS 가 시작하자마자 "수렴"으로 착각하고 멈춘다)
    def loss(v):
        w = v * scale / s
        p = sigmoid(Xs @ v)
        err = p - y
        diff = (w - w0) / REG_UNIT
        value = np.mean(err ** 2) + lam * np.sum(diff ** 2)
        grad = (2.0 / len(y)) * (Xs.T @ (err * p * (1.0 - p))) + 2.0 * lam * diff * scale / (REG_UNIT * s)
        return value, grad

    res = minimize(loss, w0 * s / scale, jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-14})
    return res.x * scale / s


def tune(X, y, gid, w0, seed=0):
    """눈금 K 를 보정하고 lam 을 검증 오차로 고른 뒤, 전체 데이터로 다시 맞춘 가중치(dict)를 돌려준다.
    신호가 너무 약해서 K 를 정할 수 없으면 None."""
    rng = np.random.default_rng(seed)
    games = np.unique(gid)
    rng.shuffle(games)
    val = np.isin(gid, games[: max(1, int(len(games) * 0.2))])
    tr = ~val

    k_train = calibrate_scale(w0, X[tr], y[tr])
    print(f"눈금 K = {k_train:.0f}  (현재 가중치 기준. 클수록 '점수가 승패를 덜 말해 준다'는 뜻)")
    if k_train > 0.8 * K_MAX:
        print("!! K 가 탐색 범위 끝에 붙었습니다. 현재 평가 점수가 자가대전 승패를 거의 설명하지 못한다는 뜻이라,")
        print("   가중치 크기가 임의로 정해져 학습 결과를 믿을 수 없습니다. 이 라운드는 학습을 건너뜁니다.")
        print("   (데이터는 쌓이므로 다음 라운드에 합쳐서 다시 시도합니다. GAMES_PER_ROUND / TIME_MS 를 늘리면 도움이 됩니다.)")
        return None
    print(f"검증 오차 기준선: 항상 0.5 예측 {np.mean((0.5 - y[val]) ** 2):.4f} | "
          f"현재 가중치 (K=200 가정) {mse(w0, X[val], y[val], 200.0):.4f} | 현재 가중치 (보정한 K) {mse(w0, X[val], y[val], k_train):.4f}")
    rows = []
    for lam in LAMBDAS:
        w = fit(X[tr], y[tr], w0, lam, k_train)
        rows.append({"lambda": lam, "train_mse": mse(w, X[tr], y[tr], k_train), "val_mse": mse(w, X[val], y[val], k_train)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    best_lam = float(table.loc[table["val_mse"].idxmin(), "lambda"])
    print("선택한 lambda:", best_lam)

    k_all = calibrate_scale(w0, X, y)
    w = fit(X, y, w0, best_lam, k_all)
    w = np.clip(np.round(w), -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
    return dict(zip(NAMES, w.astype(int).tolist()))
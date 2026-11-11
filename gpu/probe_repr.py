"""입력 표현 진단: 강화학습 신경망(합성곱 c64b6)이 규칙상 정확히 계산되는 전술 사실을 지금 입력만으로 배울 수 있는지 본다.

정답 (둘 차례인 쪽 기준, 규칙으로 정확히 계산):
  win1   지금 움직일 말(mc)로 바로 골에 갈 수 있나
  threat 상대가 다음에 움직일 말(oc)이 (판이 그대로면) 바로 골에 갈 수 있나
  lose   수마다: 그 수를 두면 상대(oc)가 바로 이기나  (196칸, 둘 수 있는 수에서만 평가)
  reach  mc 가 걸어갈 수 있는 타일 (49칸)
입력:
  base   지금 강화학습 입력 (9장: 타일/말/차례/골/1)
  plus   base + 규칙으로 계산한 4장: mc 가 걸을 수 있는 칸, oc 가 걸을 수 있는 칸,
         mc 가 거기서 골로 미끄러질 수 있는 칸, oc 가 거기서 골로 미끄러질 수 있는 칸

사용: python probe_repr.py data <engine.jsonl> <out.npz>      (데이터 + 정답 만들기)
      python probe_repr.py train <data.npz> <base|plus> <steps> (학습하고 검증 정확도 출력)
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import alphazero as az
import slider_jax as sj

N = sj.N


def bits(x):
    return ((int(x) >> np.arange(N)) & 1).astype(bool)


# ------------------------------------------------------------------ 정답 계산
def walk(tiles, me, opp, piece):
    s = jnp.argmax(piece)
    blocked = (me | opp) & ~(jnp.arange(N) == s)
    return sj.reach(tiles, blocked, s)


def goal_from(tiles, walkable):
    """걸을 수 있는 칸 중, 거기서 빈 이웃 쪽으로 미끄러지면 골에 멈추는 칸"""
    tp = jnp.concatenate([tiles, jnp.array([True])])
    empty_nb = ~tp[sj.NB[:N]]
    return (walkable[:, None] & empty_nb & (sj.slide_dest(tiles) == sj.GOAL)).any(-1)


def labels_one(tiles, me, opp, mc, oc):
    valid, s = sj.legal_turn(tiles, me, opp, mc)
    dest = sj.slide_dest(tiles).reshape(sj.NUM_TURN_MOVES)
    win1 = (valid & (dest == sj.GOAL)).any()
    ov, _ = sj.legal_turn(tiles, opp, me, oc)
    threat = (ov & (dest == sj.GOAL)).any()

    def after(a):
        t2, m2, dst = sj.apply_turn(tiles, me, s, a)
        ov2, _ = sj.legal_turn(t2, opp, m2, oc)
        d2 = sj.slide_dest(t2).reshape(sj.NUM_TURN_MOVES)
        return (ov2 & (d2 == sj.GOAL)).any() & (dst != sj.GOAL)

    lose = jax.vmap(after)(jnp.arange(sj.NUM_TURN_MOVES)) & valid
    rmc = walk(tiles, me, opp, mc)
    roc = walk(tiles, opp, me, oc)
    extra = jnp.stack([rmc, roc, goal_from(tiles, rmc), goal_from(tiles, roc)])
    return win1, threat, lose, valid, rmc, extra


labels = jax.jit(jax.vmap(labels_one))


def random_positions(n, seed=0, batch=1024, steps=240):
    """무작위로 둔 판의 국면 (이기면 처음 배치로 다시). 바로 이기는 국면도 들어가게 한다."""
    rng = np.random.default_rng(seed)
    base = [np.asarray(x, bool) for x in (sj.START_TILES, sj.CYCLE_P0, sj.CYCLE_P1, sj.CYCLE_C0, sj.CYCLE_C1)]

    @jax.jit
    def step(st, u):
        def one(t, m, o, mc, oc, u):
            valid, s = sj.legal_turn(t, m, o, mc)
            n = valid.sum()
            k = jnp.floor(u * jnp.maximum(n, 1)).astype(jnp.int32)
            a = sj.nth_true(valid, jnp.clip(k, 0, jnp.maximum(n - 1, 0)))
            t2, m2, dst = sj.apply_turn(t, m, s, a)
            has = n > 0
            nt, nm = jnp.where(has, t2, t), jnp.where(has, m2, m)
            return nt, o, nm, oc, m & ~mc, has & (dst == sj.GOAL)
        return jax.vmap(one)(*st, u)

    st = [np.repeat(b[None], batch, 0) for b in base]
    out = [[] for _ in range(5)]
    got = 0
    while got < n:
        for t in range(steps):
            nxt = step(tuple(jnp.asarray(x) for x in st), jnp.asarray(rng.random(batch), jnp.float32))
            win = np.asarray(nxt[5])
            st = [np.array(x) for x in nxt[:5]]
            for j in range(5):
                st[j][win] = base[j]
            pick = rng.random(batch) < 0.05
            for j in range(5):
                out[j].append(st[j][pick])
            got += int(pick.sum())
            if got >= n:
                break
    return [np.concatenate(o)[:n] for o in out]


def make_data(eng_path, out_path):
    t0 = time.time()
    rows = [json.loads(l) for l in open(eng_path)]
    eng = [np.stack([bits(r[k]) for r in rows]) for k in ("tiles", "me", "opp", "mc", "oc")]
    rnd = random_positions(len(rows) // 2)
    src = np.concatenate([np.zeros(len(rows), np.int8), np.ones(len(rnd[0]), np.int8)])
    st = [np.concatenate([a, b]) for a, b in zip(eng, rnd)]
    res = [[] for _ in range(6)]
    for i in range(0, len(src), 2048):
        out = labels(*[jnp.asarray(x[i:i + 2048]) for x in st])
        for j in range(6):
            res[j].append(np.asarray(out[j]))
    win1, threat, lose, valid, rmc, extra = [np.concatenate(r) for r in res]
    obs = np.concatenate(st, axis=1).astype(np.uint8)
    np.savez_compressed(out_path, obs=obs, extra=extra.astype(np.uint8), win1=win1, threat=threat, lose=lose, valid=valid, reach=rmc, src=src)
    print(f"{len(src)} 국면 (엔진 {len(rows)}, 무작위 {len(rnd[0])}) {time.time() - t0:.0f}초")
    for name, m in (("엔진", src == 0), ("무작위", src == 1)):
        lv = lose[m][valid[m]]
        print(f"  {name}: win1 {win1[m].mean():.3f}  threat {threat[m].mean():.3f}  "
              f"수 중 바로 지는 수 {lv.mean():.3f}  둘 수 있는 수 평균 {valid[m].sum(1).mean():.1f}  걸을 수 있는 칸 평균 {rmc[m].sum(1).mean():.1f}")


# ------------------------------------------------------------------ 진단 신경망 (강화학습과 같은 몸통)
def init(key, in_planes, channels=64, blocks=6):
    p = az.init_conv_params(key, channels, blocks)
    k = jax.random.split(key, 4)
    p["stem"]["w"] = jax.random.normal(k[0], (3, 3, in_planes, channels)) * np.sqrt(2.0 / (9 * in_planes))
    p["lose"] = {"w": jax.random.normal(k[1], (1, 1, channels, 4)) * 0.05, "b": jnp.zeros(4)}
    p["reach"] = {"w": jax.random.normal(k[2], (1, 1, channels, 1)) * 0.05, "b": jnp.zeros(1)}
    p["glob"] = {"w": jax.random.normal(k[3], (channels, 2)) * 0.05, "b": jnp.zeros(2)}
    for drop in ("p_conv", "policy", "v_conv", "v1", "value"):
        p.pop(drop)
    return p


def forward(p, obs, extra):
    x = az._planes(obs.astype(jnp.float32))
    if extra is not None:
        x = jnp.concatenate([x, extra.astype(jnp.float32).transpose(0, 2, 1).reshape(-1, 7, 7, 4)], -1)
    h = jax.nn.relu(az._conv(x, p["stem"]))
    for b in p["blocks"]:
        h = jax.nn.relu(h + az._conv(jax.nn.relu(az._conv(h, b["c1"])), b["c2"]))
    n = h.shape[0]
    lose = az._conv(h, p["lose"]).reshape(n, N * 4)  # 칸 c, 방향 d -> c*4+d (모델 출력 [7,7,4] 를 펼친 순서와 같음)
    reach = az._conv(h, p["reach"]).reshape(n, N)
    g = h.max(axis=(1, 2)) @ p["glob"]["w"] + p["glob"]["b"]
    return g[:, 0], g[:, 1], lose, reach


def run(path, variant, steps, batch=128, seed=0):
    d = np.load(path)
    use = variant == "plus"
    n = len(d["src"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    te, tr = perm[:4000], perm[4000:]
    p = init(jax.random.PRNGKey(seed), 13 if use else 9)
    opt = optax.adamw(optax.cosine_decay_schedule(1e-3, steps), weight_decay=1e-4)
    os_ = opt.init(p)

    def loss(p, o, e, w1, th, lo, va, re):
        a, b, l, r = forward(p, o, e if use else None)
        bce = optax.sigmoid_binary_cross_entropy
        ll = (bce(l, lo) * va).sum() / jnp.maximum(va.sum(), 1)
        return bce(a, w1).mean() + bce(b, th).mean() + ll + bce(r, re).mean()

    @jax.jit
    def step(p, os_, *xs):
        g = jax.grad(loss)(p, *xs)
        u, os_ = opt.update(g, os_, p)
        return optax.apply_updates(p, u), os_

    pred = jax.jit(lambda p, o, e: forward(p, o, e if use else None))

    def batch_of(idx):
        f = lambda k, t=jnp.float32: jnp.asarray(d[k][idx], t)
        return f("obs"), f("extra"), f("win1"), f("threat"), f("lose"), f("valid"), f("reach")

    def evaluate():
        out = [[] for _ in range(4)]
        for i in range(0, len(te), 500):
            idx = te[i:i + 500]
            r = pred(p, jnp.asarray(d["obs"][idx]), jnp.asarray(d["extra"][idx]))
            for j in range(4):
                out[j].append(np.asarray(r[j]))
        a, b, l, r = [np.concatenate(o) for o in out]
        src, va = d["src"][te], d["valid"][te]
        res = {}
        for name, m in (("엔진", src == 0), ("무작위", src == 1)):
            w1, th, lo, re = d["win1"][te][m], d["threat"][te][m], d["lose"][te][m], d["reach"][te][m]
            v = va[m]
            lp, lt = (l[m] > 0)[v], lo[v]
            tp = (lp & lt).sum()
            res[name] = dict(
                win1=((a[m] > 0) == w1).mean(), threat=((b[m] > 0) == th).mean(),
                lose_acc=(lp == lt).mean(), lose_recall=tp / max(lt.sum(), 1), lose_prec=tp / max(lp.sum(), 1),
                # 한 국면에서 "바로 지는 수" 판단을 모든 수에 대해 다 맞힌 비율 (실제 대국에 필요한 것)
                lose_all=np.mean([((l[m][i] > 0) == lo[i])[v[i]].all() for i in range(len(v))]),
                reach_all=(((r[m] > 0) == re).all(1)).mean())
        return res

    t0 = time.time()
    for s in range(1, steps + 1):
        p, os_ = step(p, os_, *batch_of(rng.choice(tr, batch)))
        if s % max(steps // 6, 1) == 0 or s == steps:
            res = evaluate()
            line = " | ".join(f"{k} " + " ".join(f"{m}={v:.3f}" for m, v in r.items()) for k, r in res.items())
            print(f"[{variant} {s}/{steps} {time.time() - t0:.0f}s] {line}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "data":
        make_data(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))

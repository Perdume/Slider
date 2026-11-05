"""무작위 대국 속도 (초당 처리한 수 = ply).  python bench.py [배치 크기 ...]"""
import os
import sys
import time

import jax
import jax.numpy as jnp

import slider_jax as sj

MAX = int(os.environ.get("MAX_PLIES", sj.DEFAULT_MAX_PLIES))
print("device:", jax.devices(), "최대 수(안전장치):", MAX)


@jax.jit
def run(keys):
    return sj.play_batch(sj.random_policy, keys, max_plies=MAX)


for B in [int(x) for x in sys.argv[1:]] or [1024, 8192, 32768]:
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    jax.block_until_ready(run(keys))  # 컴파일
    t0 = time.time()
    w, p, e = jax.block_until_ready(run(jax.random.split(jax.random.PRNGKey(1), B)))
    s = time.time() - t0
    plies = int(p.sum())
    print(f"배치 {B:6d}: {B}판 {plies}수 {s:.2f}s -> {plies / s:,.0f} plies/s (승부 난 판 {(w >= 0).sum()}, 반복 무승부 {(e & (w < 0)).sum()}, 안 끝난 판 {(~e).sum()}, 평균 {plies / B:.0f}수)")

"""무작위 대국 속도 (초당 처리한 수 = ply).  python bench.py [배치 크기 ...]"""
import sys
import time

import jax
import jax.numpy as jnp

import slider_jax as sj

print("device:", jax.devices())


@jax.jit
def run(keys):
    w, p = jax.vmap(lambda k: sj.play_game(sj.random_policy, k))(keys)
    return w, p


for B in [int(x) for x in sys.argv[1:]] or [256, 1024, 4096]:
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    jax.block_until_ready(run(keys))  # 컴파일
    t0 = time.time()
    w, p = jax.block_until_ready(run(jax.random.split(jax.random.PRNGKey(1), B)))
    s = time.time() - t0
    plies = int(p.sum())
    print(f"배치 {B:6d}: {B}판 {plies}수 {s:.2f}s -> {plies / s:,.0f} plies/s (승부 난 판 {(w >= 0).sum()})")

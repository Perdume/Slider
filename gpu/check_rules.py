"""Rust 규칙과 JAX 규칙이 같은지 확인한다.  python check_rules.py <rules_dump 실행 파일>"""
import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

import slider_jax as sj

DUMP = sys.argv[1]


def run(*args):
    return subprocess.run([DUMP, *map(str, args)], check=True, capture_output=True, text=True).stdout.splitlines()


def bits(x):
    return np.array([(x >> i) & 1 for i in range(sj.N)], bool)


def check_perft(depth=6):
    expect = {int(l.split()[1]): int(l.split()[2]) for l in run("perft", depth)}
    legal_b = jax.jit(jax.vmap(sj.legal))
    apply_b = jax.jit(jax.vmap(sj.apply_move))
    tiles, me, opp = (np.asarray(x)[None] for x in (sj.START_TILES, sj.START_P0, sj.START_P1))
    wins_total = 0
    for d in range(1, depth + 1):
        valid, srcs = (np.asarray(x) for x in legal_b(tiles, me, opp))
        pi, mi = np.nonzero(valid)
        npass = np.nonzero(valid.sum(1) == 0)[0]
        nt, nm, dst = (np.asarray(x) for x in apply_b(tiles[pi], me[pi], srcs[pi], mi.astype(np.int32)))
        win = dst == sj.GOAL
        wins_total += int(win.sum())
        keep = ~win
        tiles2 = np.concatenate([nt[keep], tiles[npass]])
        me2 = np.concatenate([opp[pi][keep], opp[npass]])
        opp2 = np.concatenate([nm[keep], me[npass]])
        tiles, me, opp = tiles2, me2, opp2
        got = len(tiles) + wins_total
        print(f"perft {d}: Rust {expect[d]}  JAX {got}  {'OK' if got == expect[d] else 'MISMATCH'}")
        assert got == expect[d]


def check_positions(games=300):
    rows = [json.loads(l) for l in run("positions", games, 7)]
    tiles = np.stack([bits(r["tiles"]) for r in rows])
    me = np.stack([bits(r["me"]) for r in rows])
    opp = np.stack([bits(r["opp"]) for r in rows])
    valid, srcs = (np.asarray(x) for x in jax.jit(jax.vmap(sj.legal))(tiles, me, opp))
    dests = np.asarray(jax.jit(jax.vmap(sj.slide_dest))(tiles))
    bad = 0
    for i, r in enumerate(rows):
        idx = np.nonzero(valid[i])[0]
        ints = sorted(int(sj_move(srcs[i], j, dests[i])) for j in idx)
        if ints != r["moves"]:
            bad += 1
    print(f"국면 {len(rows)}개의 가능한 수 목록: 다른 국면 {bad}개  {'OK' if bad == 0 else 'MISMATCH'}")
    assert bad == 0


def sj_move(srcs, idx, dests):
    k, c, d = idx // 196, (idx % 196) // 4, idx % 4
    return c | (d << 6) | (int(dests[c, d]) << 8) | (int(srcs[k]) << 14)


def check_games(a_max=20, b_max=20):
    expect = [tuple(map(int, l.split())) for l in run("games", a_max, b_max)]
    a = jnp.array([e[0] for e in expect])
    b = jnp.array([e[1] for e in expect])

    @jax.jit
    @jax.vmap
    def play(a, b):
        return sj.play_game(lambda valid, n, ply, key: (ply * a + b) % jnp.maximum(n, 1), jax.random.PRNGKey(0), max_plies=150)

    w, p, _ = (np.asarray(x) for x in play(a, b))
    bad = sum(1 for e, ww, pp in zip(expect, w, p) if (e[2], e[3]) != (int(ww), int(pp)))
    kinds = {k: int((w == k).sum()) for k in (-1, 0, 1)}
    early = int((p < 150).sum())
    print(f"정해진 규칙 대국 {len(expect)}판 (무승부/선승/후승 {kinds}, 150수 전에 끝난 판 {early}): 결과 다른 판 {bad}개  {'OK' if bad == 0 else 'MISMATCH'}")
    assert bad == 0


if __name__ == "__main__":
    check_perft()
    check_positions()
    check_games()

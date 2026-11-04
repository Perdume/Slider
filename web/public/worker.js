// AI 계산 전용 워커: 화면이 멈추지 않도록 탐색을 여기서 한다.
let E = null;
const ready = (async () => {
  const imports = { env: { now_ms: () => performance.now() } };
  try {
    E = (await WebAssembly.instantiateStreaming(fetch("engine.wasm"), imports)).instance.exports;
  } catch (e) {
    const buf = await (await fetch("engine.wasm")).arrayBuffer();
    E = (await WebAssembly.instantiate(buf, imports)).instance.exports;
  }
})();

onmessage = async e => {
  await ready;
  const { seq, tiles, me, opp, mc, oc, hist, time } = e.data;
  E.hist_clear();
  for (const h of hist) E.hist_add(h);
  const mv = E.think(tiles, me, opp, mc, oc, time);
  postMessage({ seq, mv, score: E.last_score(), depth: E.last_depth(), nodes: E.last_nodes() });
};

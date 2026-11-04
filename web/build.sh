#!/usr/bin/env bash
# 엔진을 WebAssembly 로 빌드하고 public/ 에 배포용 파일을 만든다.
# 필요: rustup target add wasm32-unknown-unknown
set -euo pipefail
cd "$(dirname "$0")"
(cd engine && cargo build --release --target wasm32-unknown-unknown)
mkdir -p public
cp engine/target/wasm32-unknown-unknown/release/slider_wasm.wasm public/engine.wasm
cp src/worker.js public/worker.js
{
  printf '<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
  printf '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
  printf '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}[hidden]{display:none!important}</style>\n'
  printf '</head>\n<body>\n'
  cat src/app.html
  printf '\n</body>\n</html>\n'
} > public/index.html
ls -la public

# Slider 웹판 (Vercel 배포용)

브라우저에서 AI 와 두는 Slider. AI 는 Rust 엔진을 WebAssembly 로 빌드한 것이라 **서버 없이 접속한 사람의 브라우저에서** 계산한다.
그래서 Vercel 에는 정적 파일만 올라가고 서버 비용이 없다 (Hobby 무료 플랜으로 충분).

- `public/` — 배포되는 파일 (index.html, worker.js, engine.wasm). **빌드 결과를 레포에 넣어 두므로 Vercel 은 빌드할 것이 없다.**
- `src/` — 화면 원본 (app.html, worker.js)
- `engine/` — 엔진을 wasm 으로 내보내는 얇은 Rust 크레이트 (`../rust` 엔진을 그대로 씀, 게임 설정: lmr2,hm + Tunables::game + weights/tuned.json)
- `build.sh` — wasm 빌드 + public/ 다시 만들기

## Vercel 에 올리기 (처음 한 번)
1. https://vercel.com 에 GitHub 로 로그인 → **Add New… → Project** → `Perdume/Slider` **Import**
2. **Root Directory** 를 `web` 으로 지정 (Framework Preset 은 Other, 나머지는 `web/vercel.json` 이 정함)
3. **Deploy**. 기본 배포 브랜치는 Settings → Git → Production Branch 에서 `web-vercel` 로 바꾼다
   (안 바꾸면 이 브랜치 push 는 미리보기 주소로 배포된다)

이후에는 이 브랜치에 push 할 때마다 자동으로 다시 배포된다.

## 엔진이나 화면을 고친 뒤
```bash
rustup target add wasm32-unknown-unknown   # 처음 한 번
bash web/build.sh                          # public/ 을 다시 만든다
cd web/public && python3 -m http.server 8000   # http://localhost:8000 에서 확인 (file:// 로는 wasm 을 못 읽음)
```
그다음 `web/public` 까지 커밋해서 push.

## 참고
- AI 는 한 스레드로 계산한다 (수당 1초에 깊이 20 안팎). 멀티스레드는 COOP/COEP 헤더 + wasm 스레드 빌드가 필요해서 나중 과제.
- 엔진 쪽 바뀐 점: `rust/src/clock.rs` — 브라우저에는 std 의 Instant 가 없어서, wasm 일 때만 JS 의 performance.now 로 시간을 잰다. 네이티브 빌드는 그대로.

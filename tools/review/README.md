# 대국 리뷰 웹 화면

AI 끼리 둔 판을 체스 대국 리뷰처럼 한 수씩 본다 (보드, 형세 막대/그래프, 수 분류, 기보, 자동 재생).

```bash
cd rust && cargo run --release --example game_json -- 6 300 2 11 > ../tools/review/games.json   # 판 수, 수당 ms, 스레드, seed
python3 ../tools/review/build.py ../tools/review/template.html ../tools/review/games.json ../tools/review/slider_review.html
```
만들어진 `slider_review.html` 을 브라우저로 열면 된다.

# 웹판은 web-vercel 브랜치에 있습니다

이 폴더는 rust-port 브랜치에서 Vercel 이 Root Directory(`web`)를 찾을 수 있게 두는 자리입니다.
Vercel 의 Ignored Build Step(`[ "$VERCEL_GIT_COMMIT_REF" != "web-vercel" ]`)이 이 브랜치 빌드를 건너뜁니다.
웹판 코드와 배포 설정은 `web-vercel` 브랜치의 `web/` 를 보세요.

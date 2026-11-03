#!/usr/bin/env bash
# Gradle 없이 안드로이드 SDK 도구만으로 APK 를 만든다.
#   ANDROID_HOME=/path/to/sdk ./build_apk.sh
# 필요한 것: Rust(rustup), JDK 17+, 안드로이드 SDK 의 platforms;android-34, build-tools;34.0.0, ndk;26.x
#   sdkmanager "platforms;android-34" "build-tools;34.0.0" "ndk;26.3.11579264"
# 결과: android/build/slider-train.apk (디버그 키로 서명, 직접 설치용). 앱 = 새 규칙 SPSA 튜닝 (libspsa.so)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SDK="${ANDROID_HOME:?ANDROID_HOME 에 안드로이드 SDK 경로를 지정하세요}"
BT="$(ls -d "$SDK"/build-tools/* | sort -V | tail -1)"
PLATFORM="$SDK/platforms/android-34/android.jar"
NDK="$(ls -d "$SDK"/ndk/* | sort -V | tail -1)"
CLANG="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"
TARGET=aarch64-linux-android
OUT="$HERE/build"
APP_BINS=(spsa)

for f in "$PLATFORM" "$CLANG" "$BT/aapt2" "$BT/d8" "$BT/zipalign" "$BT/apksigner"; do
  [ -e "$f" ] || { echo "없음: $f" >&2; exit 1; }
done

echo "== 1/6 Rust 실행 파일 (arm64) 빌드"
rustup target add "$TARGET" >/dev/null
(cd "$HERE/../rust" && CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CLANG" cargo build --release --target "$TARGET" $(printf -- '--bin %s ' "${APP_BINS[@]}"))

rm -rf "$OUT" && mkdir -p "$OUT/lib/arm64-v8a" "$OUT/classes"
for b in "${APP_BINS[@]}"; do
  # 앱이 설치될 때 lib*.so 만 네이티브 폴더로 풀리므로 이름을 맞춰서 넣는다 (실제로는 실행 파일)
  cp "$HERE/../rust/target/$TARGET/release/$b" "$OUT/lib/arm64-v8a/lib$b.so"
done

echo "== 2/6 매니페스트 링크 (aapt2)"
"$BT/aapt2" link -o "$OUT/base.apk" -I "$PLATFORM" --manifest "$HERE/AndroidManifest.xml" \
  --min-sdk-version 29 --target-sdk-version 33 --version-code 2 --version-name 0.2

echo "== 3/6 Java 컴파일"
# android.jar 에는 람다에 필요한 java.lang.invoke 가 없어서, 자바 API 는 JDK(--release 8)에서 가져오고 안드로이드 API 만 android.jar 에서 가져온다
javac -encoding UTF-8 --release 8 -classpath "$PLATFORM" -d "$OUT/classes" $(find "$HERE/src" -name '*.java')

echo "== 4/6 dex 변환 (d8)"
"$BT/d8" --release --min-api 29 --lib "$PLATFORM" --output "$OUT" $(find "$OUT/classes" -name '*.class')

echo "== 5/6 APK 조립 + 정렬"
(cd "$OUT" && cp base.apk unaligned.apk && zip -q -r unaligned.apk classes.dex lib)
"$BT/zipalign" -f -p 4 "$OUT/unaligned.apk" "$OUT/aligned.apk"

echo "== 6/6 서명 (디버그 키)"
KS="$HERE/debug.keystore"
if [ ! -f "$KS" ]; then
  keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias debug -keyalg RSA -keysize 2048 \
    -validity 10000 -dname "CN=Slider Debug,O=Perdume,C=KR" >/dev/null 2>&1
fi
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out "$OUT/slider-train.apk" "$OUT/aligned.apk"
"$BT/apksigner" verify "$OUT/slider-train.apk"
ls -l "$OUT/slider-train.apk"

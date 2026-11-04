//! 시간 재기. 보통은 std::time::Instant 를 쓰고, 브라우저(wasm32)에서는 std 의 Instant 가 없으므로
//! JS 가 넘겨 주는 `now_ms()` (performance.now) 로 같은 모양의 Instant 를 만든다.

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
mod web {
    use std::time::Duration;

    #[link(wasm_import_module = "env")]
    extern "C" {
        fn now_ms() -> f64;
    }

    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct Instant(f64);

    impl Instant {
        pub fn now() -> Self {
            Instant(unsafe { now_ms() })
        }
        pub fn checked_add(&self, d: Duration) -> Option<Self> {
            Some(Instant(self.0 + d.as_secs_f64() * 1000.0))
        }
        pub fn elapsed(&self) -> Duration {
            Duration::from_secs_f64(((Instant::now().0 - self.0) / 1000.0).max(0.0))
        }
    }

    impl std::ops::Add<Duration> for Instant {
        type Output = Instant;
        fn add(self, d: Duration) -> Instant {
            Instant(self.0 + d.as_secs_f64() * 1000.0)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::Instant;

//! 두 실행 파일이 같이 쓰는 명령줄 도우미.

use std::collections::HashMap;
use std::str::FromStr;

/// `--이름 값` 형식의 옵션을 읽는다.
pub fn parse_args(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if !args[i].starts_with("--") || i + 1 >= args.len() {
            return Err(format!("옵션은 '--이름 값' 형식이어야 합니다: {}", args[i]));
        }
        map.insert(args[i][2..].to_string(), args[i + 1].clone());
        i += 2;
    }
    Ok(map)
}

/// 옵션 값을 읽고, 없으면 기본값을 쓴다. 숫자가 아니면 오류.
pub fn get<T: FromStr>(opt: &HashMap<String, String>, name: &str, default: T) -> Result<T, String> {
    match opt.get(name) {
        None => Ok(default),
        Some(v) => v.parse::<T>().map_err(|_| format!("--{} 값을 읽을 수 없습니다: {}", name, v)),
    }
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// 실행 중인 작업을 깨끗하게 멈추기 위한 신호. 두 가지 중 하나라도 해당하면 멈춘다.
///  - `--stop-file 경로`: 그 파일이 생기면 멈춘다 (앱의 "중지" 버튼)
///  - `--parent-pid N`: 그 프로세스가 사라지면 멈춘다 (앱이 강제 종료돼도 자식이 고아로 남지 않게)
/// 작업 단위(한 판, 한 epoch)가 끝날 때마다 확인하므로 몇 초 안에 멈춘다.
#[derive(Clone, Debug, Default)]
pub struct Stop {
    file: Option<std::path::PathBuf>,
    parent: Option<u32>,
}

impl Stop {
    pub fn from_options(opt: &HashMap<String, String>) -> Result<Self, String> {
        Ok(Stop {
            file: opt.get("stop-file").map(std::path::PathBuf::from),
            parent: match opt.get("parent-pid") {
                Some(v) => Some(v.parse::<u32>().map_err(|_| format!("--parent-pid 값을 읽을 수 없습니다: {}", v))?),
                None => None,
            },
        })
    }

    pub fn should_stop(&self) -> bool {
        if let Some(f) = &self.file {
            if f.exists() {
                return true;
            }
        }
        if let Some(pid) = self.parent {
            if !process_alive(pid) {
                return true;
            }
        }
        false
    }

    /// 자식 프로세스에 그대로 넘길 옵션들
    pub fn child_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(f) = &self.file {
            args.push("--stop-file".to_string());
            args.push(f.display().to_string());
        }
        if let Some(p) = self.parent {
            args.push("--parent-pid".to_string());
            args.push(p.to_string());
        }
        args
    }
}

/// /proc/<pid>/stat 의 상태 글자를 보고 살아 있는지 판단한다. 폴더만 남은 좀비(Z)나 이미 죽은(X) 프로세스는 죽은 것으로 본다.
fn process_alive(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{}/stat", pid)) {
        // 형식: "pid (이름) 상태 ..." 이름에 공백/괄호가 들어갈 수 있어서 마지막 ')' 뒤를 본다
        Ok(stat) => match stat.rfind(')').and_then(|i| stat[i + 1..].trim_start().chars().next()) {
            Some(state) => state != 'Z' && state != 'X',
            None => true,
        },
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_process_is_alive_and_a_missing_pid_is_not() {
        if cfg!(any(target_os = "linux", target_os = "android")) {
            assert!(process_alive(std::process::id()));
            assert!(!process_alive(u32::MAX - 1));
        }
    }

    #[test]
    fn stop_file_triggers_stop() {
        let dir = std::env::temp_dir().join(format!("slider-stop-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("STOP");
        let mut opt = HashMap::new();
        opt.insert("stop-file".to_string(), file.display().to_string());
        let stop = Stop::from_options(&opt).unwrap();
        assert!(!stop.should_stop());
        std::fs::write(&file, "").unwrap();
        assert!(stop.should_stop());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

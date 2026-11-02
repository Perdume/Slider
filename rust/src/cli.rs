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

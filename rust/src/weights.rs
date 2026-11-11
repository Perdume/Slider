//! 평가 함수의 가중치. 평가 점수 = sum(가중치[i] * 특징[i]) 이고, 특징은 항상 "둘 차례인 쪽" 기준이다.
//!
//! JSON 형식 (이름은 NAMES 와 같아야 하고, 빠진 이름은 기본값을 쓴다): {"threat1": -150, "threat2": -400, ...}
//!
//! 특징 정의 (둘 차례인 쪽 = me, 반대편 = opp):
//!  threat1          opp 가 바로 이기는 수가 정확히 1개면 1
//!  threat2          opp 가 바로 이기는 수가 2개 이상이면 1
//!  threat2PerThreat 그 경우의 수의 개수 (2개 이상일 때만, 아니면 0)
//!  mobility         (me 의 이동 가능 수) - (opp 의 이동 가능 수)
//!  reach            (me 말들이 걸을 수 있는 타일 수) - (opp 의 것)
//!  plan2..plan5     골로 들어가는 4방향 직선 중 이기기까지 남은 수(어림)가 2..5인 직선 개수의 me - opp 차이
//!  position         (me 말들의 중앙 근접 점수 합) - (opp 의 것)
//!  tempo            항상 1 (둘 차례인 쪽의 이점)

use std::fmt;
use std::path::Path;

pub const NAMES: [&str; 11] = [
    "threat1", "threat2", "threat2PerThreat", "mobility", "reach",
    "plan2", "plan3", "plan4", "plan5", "position", "tempo",
];
pub const COUNT: usize = NAMES.len();

pub const THREAT1: usize = 0;
pub const THREAT2: usize = 1;
pub const THREAT2_PER: usize = 2;
pub const MOBILITY: usize = 3;
pub const REACH: usize = 4;
pub const PLAN_FIRST: usize = 5;
pub const POSITION: usize = 9;
pub const TEMPO: usize = 10;

// 튜닝 전에 손으로 정한 값 (Java 구현과 같다)
const DEFAULTS: [i32; COUNT] = [-150, -400, -50, 1, 2, 150, 70, 30, 12, 1, 0];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalWeights {
    pub values: [i32; COUNT],
}

impl EvalWeights {
    pub fn defaults() -> Self {
        EvalWeights { values: DEFAULTS }
    }

    pub fn of(values: [i32; COUNT]) -> Self {
        EvalWeights { values }
    }

    /// 평면 JSON 을 읽는다. 모르는 이름은 오타일 수 있어서 오류로 처리한다. 값은 반올림해서 정수로 쓴다.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let mut values = DEFAULTS;
        let bytes = json.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'"' {
                i += 1;
                continue;
            }
            match parse_entry(json, i) {
                Some((name, value, end)) => {
                    let index = NAMES.iter().position(|n| *n == name).ok_or_else(|| {
                        format!("알 수 없는 가중치 이름: {} (사용 가능: {:?})", name, NAMES)
                    })?;
                    values[index] = (value + 0.5).floor() as i32; // Java Math.round 와 같은 반올림
                    i = end;
                }
                None => i += 1,
            }
        }
        Ok(EvalWeights { values })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("가중치 파일을 읽을 수 없습니다: {}: {}", path.display(), e))?;
        Self::from_json(&text)
    }

    /// None 이거나 "default" 이면 기본값, 아니면 그 경로의 JSON.
    pub fn from_argument(arg: Option<&str>) -> Result<Self, String> {
        match arg {
            None | Some("default") => Ok(Self::defaults()),
            Some(path) => Self::load(Path::new(path)),
        }
    }

    pub fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        for i in 0..COUNT {
            s.push_str(&format!("  \"{}\": {}", NAMES[i], self.values[i]));
            s.push_str(if i + 1 < COUNT { ",\n" } else { "\n" });
        }
        s.push('}');
        s
    }
}

impl fmt::Display for EvalWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json().replace('\n', ""))
    }
}

/// `"이름" : 숫자` 하나를 pos(여는 따옴표 위치)에서 읽는다. 맞지 않으면 None.
fn parse_entry(json: &str, pos: usize) -> Option<(String, f64, usize)> {
    let bytes = json.as_bytes();
    let mut i = pos + 1;
    let name_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    if i == name_start || i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    let name = json[name_start..i].to_string();
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let num_start = i;
    while i < bytes.len() && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
        i += 1;
    }
    let value: f64 = json[num_start..i].parse().ok()?;
    Some((name, value, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_keeps_every_weight() {
        let original = EvalWeights::of([-151, -399, -49, 2, 3, 140, 66, 31, 13, 2, -4]);
        assert_eq!(EvalWeights::from_json(&original.to_json()).unwrap(), original);
    }

    #[test]
    fn missing_names_keep_defaults_and_numbers_are_rounded() {
        let w = EvalWeights::from_json("{\"mobility\": 4.6, \"tempo\": -2.4}").unwrap();
        assert_eq!(w.values[MOBILITY], 5);
        assert_eq!(w.values[TEMPO], -2);
        assert_eq!(w.values[REACH], EvalWeights::defaults().values[REACH]);
    }

    #[test]
    fn unknown_name_is_rejected() {
        assert!(EvalWeights::from_json("{\"mobilty\": 3}").is_err());
    }

    #[test]
    fn display_is_single_line_like_java() {
        let s = EvalWeights::defaults().to_string();
        assert!(!s.contains('\n'));
        assert!(s.starts_with("{  \"threat1\": -150,  \"threat2\": -400"));
    }
}

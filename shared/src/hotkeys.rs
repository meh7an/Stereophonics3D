//! Hotkey bindings and readout settings from `config.toml`.
//!
//! A binding is written as modifiers plus one key, such as `Ctrl+F3`, and resolved to a
//! Windows virtual-key code here, so the layer only polls numbers. Bindings are global:
//! a profile cannot change them.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::config::Issue;

/// Modifiers plus one key, polled with `GetAsyncKeyState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct KeyCombo {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// Windows virtual-key code of the non-modifier key.
    pub key: u16,
}

/// Named keys besides letters, digits, `F1`-`F24` and `Num0`-`Num9`.
const NAMED_KEYS: [(&str, u16); 17] = [
    ("Space", 0x20),
    ("PageUp", 0x21),
    ("PageDown", 0x22),
    ("End", 0x23),
    ("Home", 0x24),
    ("Left", 0x25),
    ("Up", 0x26),
    ("Right", 0x27),
    ("Down", 0x28),
    ("Insert", 0x2D),
    ("Delete", 0x2E),
    ("Tab", 0x09),
    ("Pause", 0x13),
    ("Plus", 0xBB),
    ("Comma", 0xBC),
    ("Minus", 0xBD),
    ("Period", 0xBE),
];
const VK_F1: u16 = 0x70;
const VK_NUMPAD0: u16 = 0x60;

/// The virtual-key code of a key name, compared case-insensitively.
fn key_code(name: &str) -> Option<u16> {
    let upper = name.to_ascii_uppercase();
    if let [byte] = upper.as_bytes()
        && byte.is_ascii_alphanumeric()
    {
        // Letters and digits are their own ASCII codes (upper case).
        return Some(u16::from(*byte));
    }
    if let Some(number) = upper.strip_prefix('F').and_then(|n| n.parse::<u16>().ok())
        && (1..=24).contains(&number)
    {
        return Some(VK_F1 + number - 1);
    }
    if let Some(digit) = upper
        .strip_prefix("NUM")
        .and_then(|n| n.parse::<u16>().ok())
        && digit <= 9
    {
        return Some(VK_NUMPAD0 + digit);
    }
    NAMED_KEYS
        .iter()
        .find(|(known, _)| known.eq_ignore_ascii_case(name))
        .map(|&(_, code)| code)
}

/// The canonical name of a virtual-key code produced by [`key_code`].
fn key_name(code: u16) -> String {
    match code {
        0x30..=0x39 | 0x41..=0x5A => char::from(code as u8).to_string(),
        VK_NUMPAD0..=0x69 => format!("Num{}", code - VK_NUMPAD0),
        VK_F1..=0x87 => format!("F{}", code - VK_F1 + 1),
        _ => NAMED_KEYS
            .iter()
            .find(|&&(_, known)| known == code)
            .map_or_else(|| format!("0x{code:02X}"), |(name, _)| (*name).to_owned()),
    }
}

impl FromStr for KeyCombo {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let mut combo = KeyCombo {
            ctrl: false,
            alt: false,
            shift: false,
            key: 0,
        };
        let parts: Vec<&str> = text.split('+').map(str::trim).collect();
        let Some((key, modifiers)) = parts.split_last() else {
            return Err("empty key combination".to_owned());
        };
        for modifier in modifiers {
            let flag = match modifier.to_ascii_lowercase().as_str() {
                "ctrl" | "control" => &mut combo.ctrl,
                "alt" => &mut combo.alt,
                "shift" => &mut combo.shift,
                _ => return Err(format!("unknown modifier `{modifier}` in `{text}`")),
            };
            *flag = true;
        }
        combo.key = key_code(key).ok_or_else(|| format!("unknown key `{key}` in `{text}`"))?;
        Ok(combo)
    }
}

impl fmt::Display for KeyCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (held, name) in [
            (self.ctrl, "Ctrl+"),
            (self.alt, "Alt+"),
            (self.shift, "Shift+"),
        ] {
            if held {
                f.write_str(name)?;
            }
        }
        f.write_str(&key_name(self.key))
    }
}

impl TryFrom<String> for KeyCombo {
    type Error = String;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

impl From<KeyCombo> for String {
    fn from(combo: KeyCombo) -> Self {
        combo.to_string()
    }
}

/// What each hotkey does; also the order bindings are polled and reported in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Action {
    ToggleStereo,
    ParallaxDown,
    ParallaxUp,
    ConvergenceDown,
    ConvergenceUp,
    SwapEyes,
    CycleMatrix,
    SaveProfile,
}

impl Action {
    pub const ALL: [Action; 8] = [
        Self::ToggleStereo,
        Self::ParallaxDown,
        Self::ParallaxUp,
        Self::ConvergenceDown,
        Self::ConvergenceUp,
        Self::SwapEyes,
        Self::CycleMatrix,
        Self::SaveProfile,
    ];

    /// The binding's key in `[hotkeys]`.
    pub fn name(self) -> &'static str {
        match self {
            Self::ToggleStereo => "toggle",
            Self::ParallaxDown => "parallax_down",
            Self::ParallaxUp => "parallax_up",
            Self::ConvergenceDown => "convergence_down",
            Self::ConvergenceUp => "convergence_up",
            Self::SwapEyes => "swap_eyes",
            Self::CycleMatrix => "cycle_matrix",
            Self::SaveProfile => "save",
        }
    }
}

/// `[hotkeys]`: one key combination per action. Defaults follow 3D Vision's layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct HotkeySettings {
    pub toggle: KeyCombo,
    pub parallax_down: KeyCombo,
    pub parallax_up: KeyCombo,
    pub convergence_down: KeyCombo,
    pub convergence_up: KeyCombo,
    pub swap_eyes: KeyCombo,
    pub cycle_matrix: KeyCombo,
    pub save: KeyCombo,
}

const fn ctrl(key: u16) -> KeyCombo {
    KeyCombo {
        ctrl: true,
        alt: false,
        shift: false,
        key,
    }
}

impl Default for HotkeySettings {
    fn default() -> Self {
        Self {
            toggle: ctrl(b'T' as u16),
            parallax_down: ctrl(VK_F1 + 2),
            parallax_up: ctrl(VK_F1 + 3),
            convergence_down: ctrl(VK_F1 + 4),
            convergence_up: ctrl(VK_F1 + 5),
            swap_eyes: ctrl(VK_F1 + 6),
            cycle_matrix: ctrl(VK_F1 + 7),
            save: ctrl(VK_F1 + 10),
        }
    }
}

impl HotkeySettings {
    pub fn binding(&self, action: Action) -> KeyCombo {
        match action {
            Action::ToggleStereo => self.toggle,
            Action::ParallaxDown => self.parallax_down,
            Action::ParallaxUp => self.parallax_up,
            Action::ConvergenceDown => self.convergence_down,
            Action::ConvergenceUp => self.convergence_up,
            Action::SwapEyes => self.swap_eyes,
            Action::CycleMatrix => self.cycle_matrix,
            Action::SaveProfile => self.save,
        }
    }

    /// One issue per binding that repeats an earlier one.
    pub fn validate(&self) -> Vec<Issue> {
        let mut issues = Vec::new();
        for (index, &action) in Action::ALL.iter().enumerate() {
            let combo = self.binding(action);
            if let Some(&earlier) = Action::ALL[..index]
                .iter()
                .find(|&&other| self.binding(other) == combo)
            {
                issues.push(Issue {
                    field: format!("hotkeys.{}", action.name()),
                    message: format!("{combo} is already bound to hotkeys.{}", earlier.name()),
                });
            }
        }
        issues
    }
}

/// When the on-screen readout is drawn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReadoutMode {
    /// After a change, for `seconds`, and while a configuration error is pending.
    #[default]
    Auto,
    Always,
    Off,
}

/// `[readout]`: the compositor's on-screen readout.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReadoutSettings {
    pub mode: ReadoutMode,
    /// How long `auto` shows the readout after a change.
    pub seconds: f32,
}

const MAX_READOUT_SECONDS: f32 = 60.0;

impl Default for ReadoutSettings {
    fn default() -> Self {
        Self {
            mode: ReadoutMode::Auto,
            seconds: 3.0,
        }
    }
}

impl ReadoutSettings {
    pub fn validate(&self) -> Vec<Issue> {
        let seconds = self.seconds;
        if seconds.is_finite() && seconds > 0.0 && seconds <= MAX_READOUT_SECONDS {
            return Vec::new();
        }
        vec![Issue {
            field: "readout.seconds".to_owned(),
            message: format!("must be in (0, {MAX_READOUT_SECONDS}], found {seconds}"),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_combo_parses_modifiers_and_key_kinds() {
        let combo: KeyCombo = "ctrl+Shift+f11".parse().expect("parse");
        assert_eq!(
            combo,
            KeyCombo {
                ctrl: true,
                alt: false,
                shift: true,
                key: 0x7A,
            }
        );
        assert_eq!("Alt+t".parse::<KeyCombo>().map(|c| c.key), Ok(0x54));
        assert_eq!("Num7".parse::<KeyCombo>().map(|c| c.key), Ok(0x67));
        assert_eq!("PageDown".parse::<KeyCombo>().map(|c| c.key), Ok(0x22));
        assert_eq!("5".parse::<KeyCombo>().map(|c| c.key), Ok(0x35));
    }

    #[test]
    fn test_key_combo_rejects_unknown_parts() {
        for text in [
            "",
            "Ctrl+",
            "Hyper+T",
            "Ctrl+F25",
            "Ctrl+Num10",
            "Ctrl+Escape",
        ] {
            assert!(text.parse::<KeyCombo>().is_err(), "{text} parsed");
        }
    }

    #[test]
    fn test_key_combo_display_round_trips_every_default() {
        let defaults = HotkeySettings::default();
        for action in Action::ALL {
            let combo = defaults.binding(action);
            assert_eq!(combo.to_string().parse::<KeyCombo>(), Ok(combo));
        }
        assert_eq!(defaults.toggle.to_string(), "Ctrl+T");
        assert_eq!(defaults.parallax_down.to_string(), "Ctrl+F3");
        assert_eq!(defaults.save.to_string(), "Ctrl+F11");
        assert_eq!(
            "Alt+Num0".parse::<KeyCombo>().map(|c| c.to_string()),
            Ok("Alt+Num0".to_owned())
        );
    }

    #[test]
    fn test_hotkeys_validate_reports_duplicates_once() {
        let mut hotkeys = HotkeySettings::default();
        assert!(hotkeys.validate().is_empty());
        hotkeys.save = hotkeys.toggle;
        let issues = hotkeys.validate();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].field, "hotkeys.save");
        assert!(issues[0].message.contains("hotkeys.toggle"));
    }

    #[test]
    fn test_readout_validate_bounds_seconds() {
        assert!(ReadoutSettings::default().validate().is_empty());
        for seconds in [0.0, -1.0, 61.0, f32::NAN] {
            let readout = ReadoutSettings {
                seconds,
                ..ReadoutSettings::default()
            };
            assert_eq!(readout.validate().len(), 1, "{seconds}");
        }
    }
}

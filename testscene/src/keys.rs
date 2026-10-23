//! Scripted key presses through `SendInput`, so the layer's hotkeys can be tested without a
//! person at the keyboard. A press holds the combination down for exactly one frame: the
//! layer polls key state at present, after the frame's keys went down and before they
//! come up again.

use anyhow::{Result, bail};
use s3d_shared::hotkeys::KeyCombo;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT, KEYEVENTF_KEYUP, SendInput,
    VIRTUAL_KEY, VK_CONTROL, VK_MENU, VK_SHIFT,
};

/// One key combination held during one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Press {
    pub frame: u32,
    pub combo: KeyCombo,
}

/// Parses `<frame>:<combination>`, e.g. `30:Ctrl+F4`.
pub fn parse_press(text: &str) -> Result<Press, String> {
    let (frame, combo) = text
        .split_once(':')
        .ok_or_else(|| format!("expected <frame>:<keys>, got {text:?}"))?;
    Ok(Press {
        frame: frame
            .parse()
            .map_err(|_| format!("invalid frame in {text:?}"))?,
        combo: combo.parse()?,
    })
}

/// The combination's keys, modifiers first.
fn keys(combo: KeyCombo) -> Vec<VIRTUAL_KEY> {
    [
        (combo.ctrl, VK_CONTROL),
        (combo.alt, VK_MENU),
        (combo.shift, VK_SHIFT),
    ]
    .into_iter()
    .filter_map(|(held, key)| held.then_some(key))
    .chain([VIRTUAL_KEY(combo.key)])
    .collect()
}

/// Presses the combination's keys, modifiers first.
pub fn down(combo: KeyCombo) -> Result<()> {
    send(keys(combo).into_iter(), KEYBD_EVENT_FLAGS(0))
}

/// Releases the combination's keys, the main key first.
pub fn up(combo: KeyCombo) -> Result<()> {
    send(keys(combo).into_iter().rev(), KEYEVENTF_KEYUP)
}

fn send(keys: impl Iterator<Item = VIRTUAL_KEY>, flags: KEYBD_EVENT_FLAGS) -> Result<()> {
    let inputs: Vec<INPUT> = keys
        .map(|key| INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: key,
                    dwFlags: flags,
                    ..Default::default()
                },
            },
        })
        .collect();
    // SAFETY: a slice of fully initialised keyboard inputs of the size SendInput expects.
    let sent = unsafe { SendInput(&inputs, size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        bail!("SendInput delivered {sent} of {} key events", inputs.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_press_reads_frame_and_combination() {
        let press = parse_press("30:Ctrl+F4").expect("parse");
        assert_eq!(press.frame, 30);
        assert_eq!(press.combo.to_string(), "Ctrl+F4");
        assert!(parse_press("Ctrl+F4").is_err());
        assert!(parse_press("x:Ctrl+F4").is_err());
        assert!(parse_press("3:Hyper+F4").is_err());
    }

    #[test]
    fn test_keys_put_modifiers_before_the_key() {
        let combo: KeyCombo = "Ctrl+Shift+T".parse().expect("parse");
        assert_eq!(
            keys(combo),
            [VK_CONTROL, VK_SHIFT, VIRTUAL_KEY(u16::from(b'T'))]
        );
    }
}

//! Live stereo controls: hotkeys polled once per present, in-memory adjustments layered
//! over the hot-reloaded settings, save-to-profile, and the readout's text.
//!
//! The files win: a reload that changes the effective settings clears every adjustment
//! except the stereo toggle, which has no file value. Saving writes the current values
//! into the profile, so the reload that follows changes nothing on screen.

use std::{
    path::PathBuf,
    sync::{Mutex, PoisonError, RwLock},
    time::{Duration, Instant},
};

use s3d_shared::{
    AnaglyphPreset, Diagnostic, MatrixChoice, Settings, Snapshot, StereoParams,
    StereoPushConstants,
    hotkeys::{Action, KeyCombo, ReadoutMode},
    settings::{ProfileUpdate, save_profile},
};
use windows::Win32::{
    System::Threading::GetCurrentProcessId,
    UI::{
        Input::KeyboardAndMouse::{GetAsyncKeyState, VK_CONTROL, VK_MENU, VK_SHIFT},
        WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId},
    },
};

use crate::{activation, dispatch::write};

/// Parallax change per key press, in percentage points of screen width.
const PARALLAX_STEP: f32 = 0.25;
/// Convergence is scaled by this per key press, so steps feel even at any depth.
const CONVERGENCE_FACTOR: f32 = 1.1;
const MIN_CONVERGENCE: f32 = 0.01;
/// Upper bound of `max_parallax_percent`, as in the configuration checks.
const MAX_PARALLAX_PERCENT: f32 = 10.0;

/// Hotkey changes on top of the loaded settings; `None` keeps the file's value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjustments {
    pub enabled: bool,
    pub max_parallax_percent: Option<f32>,
    pub convergence: Option<f32>,
    pub swap_eyes: Option<bool>,
    pub matrix: Option<MatrixChoice>,
}

impl Default for Adjustments {
    fn default() -> Self {
        Self {
            enabled: true,
            max_parallax_percent: None,
            convergence: None,
            swap_eyes: None,
            matrix: None,
        }
    }
}

/// The values the layer renders with: settings plus adjustments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Live {
    pub enabled: bool,
    pub stereo: StereoParams,
    pub matrix: MatrixChoice,
}

impl Live {
    /// The stereo block to push; mono (`eye_sign` 0) while stereo is toggled off.
    pub fn push_constants(&self) -> StereoPushConstants {
        let eye = if self.enabled { self.stereo.eye } else { None };
        self.stereo.push_constants(eye)
    }

    /// What the readout calls the current mode.
    fn mode(&self) -> String {
        match (self.enabled, self.stereo.eye) {
            (false, _) => "stereo off".to_owned(),
            (true, Some(eye)) => format!("single-eye {}", eye.name()),
            (true, None) => "mono".to_owned(),
        }
    }
}

impl Adjustments {
    pub fn apply(&self, settings: &Settings) -> Live {
        let stereo = &settings.stereo;
        Live {
            enabled: self.enabled,
            stereo: StereoParams {
                max_parallax_percent: self
                    .max_parallax_percent
                    .unwrap_or(stereo.max_parallax_percent),
                convergence: self.convergence.unwrap_or(stereo.convergence),
                swap_eyes: self.swap_eyes.unwrap_or(stereo.swap_eyes),
                eye: stereo.eye,
            },
            matrix: self.matrix.unwrap_or(settings.anaglyph.matrix),
        }
    }

    /// Applies one hotkey to `live`, the values currently in effect. Saving is not an
    /// adjustment and leaves them unchanged.
    pub fn act(&mut self, action: Action, live: &Live) {
        let stereo = &live.stereo;
        match action {
            Action::ToggleStereo => self.enabled = !self.enabled,
            Action::ParallaxDown | Action::ParallaxUp => {
                let step = if action == Action::ParallaxUp {
                    PARALLAX_STEP
                } else {
                    -PARALLAX_STEP
                };
                let value =
                    (stereo.max_parallax_percent + step).clamp(PARALLAX_STEP, MAX_PARALLAX_PERCENT);
                self.max_parallax_percent = Some(value);
            }
            Action::ConvergenceDown => {
                let value = (stereo.convergence / CONVERGENCE_FACTOR).max(MIN_CONVERGENCE);
                self.convergence = Some(value);
            }
            Action::ConvergenceUp => {
                self.convergence = Some(stereo.convergence * CONVERGENCE_FACTOR)
            }
            Action::SwapEyes => self.swap_eyes = Some(!stereo.swap_eyes),
            Action::CycleMatrix => {
                let next = match live.matrix {
                    MatrixChoice::Preset(preset) => preset.next(),
                    MatrixChoice::Custom(_) => AnaglyphPreset::ALL[0],
                };
                self.matrix = Some(MatrixChoice::Preset(next));
            }
            Action::SaveProfile => {}
            _ => {}
        }
    }

    /// Adjustments with only the stereo toggle kept, for after a reload.
    fn after_reload(&self) -> Self {
        Self {
            enabled: self.enabled,
            ..Self::default()
        }
    }
}

/// The current values written into a profile.
pub fn profile_update(live: &Live) -> ProfileUpdate {
    ProfileUpdate {
        max_parallax_percent: Some(live.stereo.max_parallax_percent),
        convergence: Some(live.stereo.convergence),
        swap_eyes: Some(live.stereo.swap_eyes),
        matrix: Some(live.matrix),
    }
}

/// Read on every pipeline bind, written only by hotkeys and reloads.
static ADJUSTMENTS: RwLock<Option<Adjustments>> = RwLock::new(None);
/// The last block pushed, to log when the pushed values change.
static PUSHED: Mutex<Option<StereoPushConstants>> = Mutex::new(None);
static STATE: Mutex<State> = Mutex::new(State::new());

/// Present-thread state: what the last poll saw and what the readout shows.
struct State {
    /// Generation of the last snapshot handled; 0 before the first poll.
    generation: u64,
    /// Settings the adjustments are layered over.
    base: Option<Settings>,
    /// Bindings held down at the last poll, in `Action::ALL` order.
    held: [bool; Action::ALL.len()],
    shown_until: Option<Instant>,
    /// A transient line, such as a save result.
    message: Option<String>,
}

impl State {
    const fn new() -> Self {
        Self {
            generation: 0,
            base: None,
            held: [false; Action::ALL.len()],
            shown_until: None,
            message: None,
        }
    }
}

fn adjustments() -> Adjustments {
    (*ADJUSTMENTS.read().unwrap_or_else(PoisonError::into_inner)).unwrap_or_default()
}

/// The values to render with under `settings`.
pub fn live(settings: &Settings) -> Live {
    adjustments().apply(settings)
}

/// The stereo block for `settings` with the adjustments applied, logging the first push
/// of each new value together with the present it belongs to.
pub fn push_constants(settings: &Settings, frame: u64) -> StereoPushConstants {
    let block = live(settings).push_constants();
    let mut pushed = PUSHED.lock().unwrap_or_else(PoisonError::into_inner);
    if *pushed != Some(block) {
        *pushed = Some(block);
        tracing::info!(
            frame,
            separation = block.separation,
            convergence = block.convergence,
            eye_sign = block.eye_sign,
            "stereo block changed"
        );
    }
    block
}

/// Handles reloads and hotkeys for present number `frame` (0-based).
pub fn poll(frame: u64) {
    let Some(snapshot) = activation::get().snapshot() else {
        return;
    };
    let Some(settings) = &snapshot.settings else {
        return;
    };
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let now = Instant::now();
    let shown_for = Duration::from_secs_f32(settings.readout.seconds);
    if snapshot.generation != state.generation {
        let first = state.generation == 0;
        state.generation = snapshot.generation;
        if state.base.as_ref() != Some(settings) {
            state.base = Some(settings.clone());
            if !first {
                let mut adjustments = write(&ADJUSTMENTS);
                *adjustments = Some(adjustments.unwrap_or_default().after_reload());
                tracing::info!(frame, generation = snapshot.generation, "settings applied");
            }
        }
        if !first {
            state.shown_until = Some(now + shown_for);
        }
    }
    if !foreground_is_ours() {
        state.held = [false; Action::ALL.len()];
        return;
    }
    for (index, &action) in Action::ALL.iter().enumerate() {
        let down = is_down(settings.hotkeys.binding(action));
        let pressed = down && !state.held[index];
        state.held[index] = down;
        if !pressed {
            continue;
        }
        state.shown_until = Some(now + shown_for);
        let current = live(settings);
        if action == Action::SaveProfile {
            let result = save_profile(&settings.profile, &profile_update(&current));
            let name = settings
                .profile
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
            state.message = Some(match result {
                Ok(()) => {
                    tracing::info!(frame, profile = %settings.profile.display(), "profile saved");
                    format!("saved to {name}")
                }
                Err(error) => {
                    tracing::warn!(frame, %error, "profile save failed");
                    format!("save failed: {error}")
                }
            });
            continue;
        }
        let mut adjustments = write(&ADJUSTMENTS);
        let mut changed = adjustments.unwrap_or_default();
        changed.act(action, &current);
        *adjustments = Some(changed);
        let after = changed.apply(settings);
        tracing::info!(
            frame,
            action = action.name(),
            enabled = after.enabled,
            parallax = after.stereo.max_parallax_percent,
            convergence = after.stereo.convergence,
            swap_eyes = after.stereo.swap_eyes,
            "hotkey"
        );
    }
}

/// A readout line and its colour (RGBA8).
pub type Line = (String, [u8; 4]);

const TEXT: [u8; 4] = [235, 235, 235, 255];
const NOTE: [u8; 4] = [140, 220, 140, 255];
const ERROR: [u8; 4] = [255, 120, 100, 255];

/// The readout's lines, or `None` when it is hidden.
pub fn readout() -> Option<Vec<Line>> {
    let snapshot = activation::get().snapshot()?;
    let settings = snapshot.settings.as_ref()?;
    let state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let recent = state
        .shown_until
        .is_some_and(|until| Instant::now() < until);
    let visible = match settings.readout.mode {
        ReadoutMode::Always => true,
        ReadoutMode::Off => false,
        _ => recent || !snapshot.diagnostics.is_empty(),
    };
    if !visible {
        return None;
    }
    Some(lines(
        &live(settings),
        state.message.as_deref().filter(|_| recent),
        &snapshot,
    ))
}

/// The readout's text for `live`, an optional transient message and the snapshot's
/// diagnostics.
fn lines(live: &Live, message: Option<&str>, snapshot: &Snapshot) -> Vec<Line> {
    let stereo = &live.stereo;
    let matrix = match live.matrix {
        MatrixChoice::Preset(preset) => preset.name(),
        MatrixChoice::Custom(_) => "custom",
    };
    let mut lines = vec![
        (format!("Stereophonics3D  {}", live.mode()), TEXT),
        (
            format!(
                "parallax {:.2} %  convergence {:.2}",
                stereo.max_parallax_percent, stereo.convergence
            ),
            TEXT,
        ),
        (
            format!(
                "matrix {matrix}  swap {}",
                if stereo.swap_eyes { "on" } else { "off" }
            ),
            TEXT,
        ),
    ];
    if let Some(message) = message {
        let color = if message.starts_with("save failed") {
            ERROR
        } else {
            NOTE
        };
        lines.push((message.to_owned(), color));
    }
    if let [first, rest @ ..] = snapshot.diagnostics.as_slice() {
        let more = if rest.is_empty() {
            String::new()
        } else {
            format!(" (+{} more)", rest.len())
        };
        // The file name alone: the full path would push the message off screen.
        let short = Diagnostic {
            file: first
                .file
                .file_name()
                .map_or_else(PathBuf::new, PathBuf::from),
            ..first.clone()
        };
        lines.push((format!("error: {short}{more}"), ERROR));
    }
    lines
}

/// Whether the foreground window belongs to this process, so keys typed into other
/// applications are ignored.
fn foreground_is_ours() -> bool {
    let mut owner = 0;
    // SAFETY: plain Win32 queries; a null window yields owner 0.
    unsafe {
        let window = GetForegroundWindow();
        GetWindowThreadProcessId(window, Some(&mut owner));
        owner == GetCurrentProcessId()
    }
}

/// Whether `combo`'s key is held with exactly its modifiers.
fn is_down(combo: KeyCombo) -> bool {
    // SAFETY: GetAsyncKeyState accepts any virtual-key code.
    let held = |key: u16| unsafe { GetAsyncKeyState(i32::from(key)) } as u16 & 0x8000 != 0;
    held(combo.key)
        && held(VK_CONTROL.0) == combo.ctrl
        && held(VK_MENU.0) == combo.alt
        && held(VK_SHIFT.0) == combo.shift
}

#[cfg(test)]
mod tests {
    use s3d_shared::{Eye, hotkeys::HotkeySettings};

    use super::*;

    fn settings() -> Settings {
        Settings {
            stereo: StereoParams {
                eye: Some(Eye::Left),
                ..StereoParams::default()
            },
            anaglyph: Default::default(),
            dump: Default::default(),
            capture: Default::default(),
            hotkeys: HotkeySettings::default(),
            readout: Default::default(),
            shaders: Default::default(),
            profile: PathBuf::from("game.toml"),
        }
    }

    fn act(adjustments: &mut Adjustments, settings: &Settings, action: Action) -> Live {
        let live = adjustments.apply(settings);
        adjustments.act(action, &live);
        adjustments.apply(settings)
    }

    #[test]
    fn test_parallax_steps_and_clamps() {
        let settings = settings();
        let mut adjustments = Adjustments::default();
        let live = act(&mut adjustments, &settings, Action::ParallaxUp);
        assert_eq!(live.stereo.max_parallax_percent, 3.25);
        for _ in 0..100 {
            act(&mut adjustments, &settings, Action::ParallaxDown);
        }
        assert_eq!(adjustments.max_parallax_percent, Some(PARALLAX_STEP));
        for _ in 0..100 {
            act(&mut adjustments, &settings, Action::ParallaxUp);
        }
        assert_eq!(adjustments.max_parallax_percent, Some(MAX_PARALLAX_PERCENT));
    }

    #[test]
    fn test_convergence_scales_both_ways() {
        let settings = settings();
        let mut adjustments = Adjustments::default();
        let up = act(&mut adjustments, &settings, Action::ConvergenceUp);
        assert!((up.stereo.convergence - 4.4).abs() < 1e-5);
        let back = act(&mut adjustments, &settings, Action::ConvergenceDown);
        assert!((back.stereo.convergence - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_toggle_pushes_mono_and_swap_negates_separation() {
        let settings = settings();
        let mut adjustments = Adjustments::default();
        let on = adjustments.apply(&settings).push_constants();
        assert_eq!(on.eye_sign, -1.0);
        let off = act(&mut adjustments, &settings, Action::ToggleStereo);
        assert_eq!(off.push_constants().eye_sign, 0.0);
        act(&mut adjustments, &settings, Action::ToggleStereo);
        let swapped = act(&mut adjustments, &settings, Action::SwapEyes);
        assert_eq!(swapped.push_constants().separation, -on.separation);
    }

    #[test]
    fn test_cycle_matrix_walks_presets_and_leaves_custom() {
        let mut settings = settings();
        let mut adjustments = Adjustments::default();
        let live = act(&mut adjustments, &settings, Action::CycleMatrix);
        assert_eq!(
            live.matrix,
            MatrixChoice::Preset(AnaglyphPreset::Dubois2009)
        );
        settings.anaglyph.matrix = MatrixChoice::Custom(AnaglyphPreset::Gray.matrix());
        let mut adjustments = Adjustments::default();
        let live = act(&mut adjustments, &settings, Action::CycleMatrix);
        assert_eq!(live.matrix, MatrixChoice::Preset(AnaglyphPreset::ALL[0]));
    }

    #[test]
    fn test_reload_keeps_only_the_toggle() {
        let settings = settings();
        let mut adjustments = Adjustments::default();
        act(&mut adjustments, &settings, Action::ToggleStereo);
        act(&mut adjustments, &settings, Action::ParallaxUp);
        let reloaded = adjustments.after_reload();
        assert_eq!(
            reloaded,
            Adjustments {
                enabled: false,
                ..Adjustments::default()
            }
        );
    }

    #[test]
    fn test_profile_update_writes_every_live_value() {
        let settings = settings();
        let mut adjustments = Adjustments::default();
        let live = act(&mut adjustments, &settings, Action::SwapEyes);
        let update = profile_update(&live);
        assert_eq!(update.swap_eyes, Some(true));
        assert_eq!(update.convergence, Some(settings.stereo.convergence));
        assert_eq!(update.matrix, Some(settings.anaglyph.matrix));
    }

    #[test]
    fn test_lines_show_values_message_and_first_error() {
        let live = Adjustments::default().apply(&settings());
        let snapshot = Snapshot {
            diagnostics: vec![
                Diagnostic {
                    file: PathBuf::from(r"C:\config\profiles\game.toml"),
                    line: Some(3),
                    field: Some("stereo.convergence".to_owned()),
                    message: "must be a positive depth, found -1".to_owned(),
                };
                2
            ],
            ..Snapshot::default()
        };
        let lines = lines(&live, Some("saved to game.toml"), &snapshot);
        let text: Vec<&str> = lines.iter().map(|(text, _)| text.as_str()).collect();
        assert_eq!(
            text,
            [
                "Stereophonics3D  single-eye left",
                "parallax 3.00 %  convergence 4.00",
                "matrix dubois-2001  swap off",
                "saved to game.toml",
                "error: game.toml:3: stereo.convergence: must be a positive depth, found -1 \
                 (+1 more)",
            ]
        );
        assert_eq!(lines[4].1, ERROR);
    }
}

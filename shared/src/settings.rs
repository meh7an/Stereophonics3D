//! Loading `config.toml` and `profiles\*.toml` into the settings one process uses.
//!
//! Loading never fails: every problem becomes a [`Diagnostic`] and the affected file keeps
//! its last valid contents (or the built-in defaults if it never had any), so a typo made
//! while a game runs cannot switch stereo off or crash the host.

use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
};

use serde::de::DeserializeOwned;
use toml::de::{DeTable, DeValue};
use toml_edit::{DocumentMut, Item};

use crate::{
    AnaglyphSettings, CaptureSettings, Config, DumpSettings, Error, MatrixChoice, Profile, Result,
    ShaderHash, ShaderOverride, StereoParams,
    config::Issue,
    hotkeys::{HotkeySettings, ReadoutSettings},
};

const CONFIG_FILE: &str = "config.toml";
const PROFILES_DIR: &str = "profiles";

/// `%APPDATA%\Stereophonics3D`, or the directory in `S3D_CONFIG_DIR` when set (tests and
/// portable setups).
///
/// # Errors
///
/// Returns [`Error::NoAppData`] when neither variable is set.
pub fn default_config_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os(crate::layer::CONFIG_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    std::env::var_os("APPDATA")
        .map(|base| PathBuf::from(base).join("Stereophonics3D"))
        .ok_or(Error::NoAppData)
}

/// `%LOCALAPPDATA%\Stereophonics3D\dumps`, or the directory in `S3D_DUMP_DIR` when set.
///
/// # Errors
///
/// Returns [`Error::NoLocalAppData`] when neither variable is set.
pub fn default_dump_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os(crate::layer::DUMP_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    std::env::var_os("LOCALAPPDATA")
        .map(|base| PathBuf::from(base).join("Stereophonics3D").join("dumps"))
        .ok_or(Error::NoLocalAppData)
}

/// `%LOCALAPPDATA%\Stereophonics3D\captures`, or the directory in `S3D_CAPTURE_DIR` when
/// set.
///
/// # Errors
///
/// Returns [`Error::NoLocalAppData`] when neither variable is set.
pub fn default_capture_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os(crate::layer::CAPTURE_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    std::env::var_os("LOCALAPPDATA")
        .map(|base| PathBuf::from(base).join("Stereophonics3D").join("captures"))
        .ok_or(Error::NoLocalAppData)
}

/// File name of the running executable, the key profiles are matched against.
pub fn current_executable_name() -> Option<String> {
    let path = std::env::current_exe().ok()?;
    Some(path.file_name()?.to_string_lossy().into_owned())
}

/// One problem found while loading, located as precisely as the file allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub file: PathBuf,
    /// 1-based line, when the problem can be tied to one.
    pub line: Option<usize>,
    /// Dotted TOML path of the offending value, when known.
    pub field: Option<String>,
    pub message: String,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.file.display())?;
        if let Some(line) = self.line {
            write!(f, ":{line}")?;
        }
        if let Some(field) = &self.field {
            write!(f, ": {field}")?;
        }
        write!(f, ": {}", self.message)
    }
}

/// Effective settings for a process with a matching profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub stereo: StereoParams,
    pub anaglyph: AnaglyphSettings,
    pub dump: DumpSettings,
    pub capture: CaptureSettings,
    pub hotkeys: HotkeySettings,
    pub readout: ReadoutSettings,
    pub shaders: BTreeMap<ShaderHash, ShaderOverride>,
    /// Profile file the settings came from.
    pub profile: PathBuf,
}

/// Everything the injector reads at run time; replaced whole on every reload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    /// `None` means no profile matches: stay pass-through.
    pub settings: Option<Settings>,
    pub diagnostics: Vec<Diagnostic>,
    /// Incremented by every load, so readers can tell a new snapshot from an old one.
    pub generation: u64,
}

/// Reads the configuration directory for one executable, remembering the last valid
/// contents of every file between loads.
#[derive(Debug)]
pub struct Loader {
    dir: PathBuf,
    executable: String,
    config: Option<Config>,
    profiles: BTreeMap<PathBuf, Profile>,
    generation: u64,
}

impl Loader {
    pub fn new(dir: &Path, executable: &str) -> Self {
        Self {
            dir: dir.to_path_buf(),
            executable: executable.to_owned(),
            config: None,
            profiles: BTreeMap::new(),
            generation: 0,
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Reads every file again and returns the resulting snapshot.
    pub fn load(&mut self) -> Snapshot {
        let mut diagnostics = Vec::new();

        let config_path = self.dir.join(CONFIG_FILE);
        if config_path.exists() {
            match read_valid::<Config>(&config_path, Config::validate) {
                Ok(config) => self.config = Some(config),
                Err(found) => diagnostics.extend(found),
            }
        } else {
            self.config = None;
        }
        let config = self.config.clone().unwrap_or_default();

        let present = profile_files(&self.dir.join(PROFILES_DIR), &mut diagnostics);
        self.profiles.retain(|path, _| present.contains(path));
        for path in &present {
            match read_valid::<Profile>(path, Profile::validate) {
                Ok(profile) => {
                    self.profiles.insert(path.clone(), profile);
                }
                Err(found) => diagnostics.extend(found),
            }
        }

        let matching: Vec<(&PathBuf, &Profile)> = self
            .profiles
            .iter()
            .filter(|(_, profile)| profile.matches(&self.executable))
            .collect();
        if let [(first, _), rest @ ..] = matching.as_slice() {
            for (other, _) in rest {
                diagnostics.push(Diagnostic {
                    file: (*other).clone(),
                    line: None,
                    field: Some("executable".to_owned()),
                    message: format!(
                        "{} is also claimed by {}, which applies instead",
                        self.executable,
                        first.display()
                    ),
                });
            }
        }
        let settings = matching.first().map(|(path, profile)| {
            let merged = config.merged(profile);
            Settings {
                stereo: merged.stereo,
                anaglyph: merged.anaglyph,
                dump: merged.dump,
                capture: merged.capture,
                hotkeys: merged.hotkeys,
                readout: merged.readout,
                shaders: profile.shaders.clone(),
                profile: (*path).clone(),
            }
        });

        self.generation += 1;
        Snapshot {
            settings,
            diagnostics,
            generation: self.generation,
        }
    }
}

/// Values the injector writes back into a profile; `None` leaves the key as it is.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProfileUpdate {
    pub max_parallax_percent: Option<f32>,
    pub convergence: Option<f32>,
    pub swap_eyes: Option<bool>,
    pub matrix: Option<MatrixChoice>,
}

/// Writes `update` into the profile at `path`, editing only the keys it sets so comments,
/// ordering and every other entry survive. The result is checked to parse as a valid
/// profile before the file is replaced.
///
/// # Errors
///
/// The file cannot be read or written, is not valid TOML, or would not be a valid profile
/// afterwards.
pub fn save_profile(path: &Path, update: &ProfileUpdate) -> Result<()> {
    let io = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    let edit_error = |message: String| Error::ProfileEdit {
        path: path.to_path_buf(),
        message,
    };
    let text = fs::read_to_string(path).map_err(io)?;
    let mut document: DocumentMut = text
        .parse()
        .map_err(|error| edit_error(format!("{error}")))?;
    let fields = [
        (
            "stereo",
            "max_parallax_percent",
            update.max_parallax_percent.map(item),
        ),
        ("stereo", "convergence", update.convergence.map(item)),
        ("stereo", "swap_eyes", update.swap_eyes.map(item)),
        ("anaglyph", "matrix", update.matrix.map(item)),
    ];
    for (table, key, value) in fields {
        let Some(value) = value else {
            continue;
        };
        let value = value.map_err(edit_error)?;
        let table = document
            .entry(table)
            .or_insert_with(|| Item::Table(toml_edit::Table::new()))
            .as_table_like_mut()
            .ok_or_else(|| edit_error(format!("`{table}` is not a table")))?;
        match (table.get_mut(key), value) {
            // Replacing only the value keeps the key, and with it the comment lines above.
            (Some(Item::Value(old)), Item::Value(mut new)) => {
                *new.decor_mut() = old.decor().clone();
                *old = new;
            }
            (_, value) => {
                table.insert(key, value);
            }
        }
    }
    let edited = document.to_string();
    let profile: Profile =
        toml::from_str(&edited).map_err(|error| edit_error(error.to_string()))?;
    if let Some(issue) = profile.validate().into_iter().next() {
        return Err(edit_error(format!("{}: {}", issue.field, issue.message)));
    }
    fs::write(path, edited).map_err(io)
}

/// `value` as a TOML item, written the way serde writes it in a whole file (preset names,
/// matrix rows, shortest float form).
fn item<T: serde::Serialize>(value: T) -> std::result::Result<Item, String> {
    #[derive(serde::Serialize)]
    struct Wrapper<T> {
        value: T,
    }
    let text = toml::to_string(&Wrapper { value }).map_err(|error| error.to_string())?;
    let mut document: DocumentMut = text.parse().map_err(|error| format!("{error}"))?;
    document
        .remove("value")
        .ok_or_else(|| "serialized value is missing".to_owned())
}

/// `*.toml` files directly inside `dir`, sorted, so "first file wins" is deterministic.
fn profile_files(dir: &Path, diagnostics: &mut Vec<Diagnostic>) -> Vec<PathBuf> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => {
            diagnostics.push(Diagnostic {
                file: dir.to_path_buf(),
                line: None,
                field: None,
                message: format!("cannot list profiles: {error}"),
            });
            return Vec::new();
        }
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "toml"))
        .collect();
    files.sort();
    files
}

/// Parses and checks one file, or returns every problem found in it.
fn read_valid<T: DeserializeOwned>(
    path: &Path,
    validate: fn(&T) -> Vec<Issue>,
) -> std::result::Result<T, Vec<Diagnostic>> {
    let diagnostic = |line, field, message| Diagnostic {
        file: path.to_path_buf(),
        line,
        field,
        message,
    };
    let text = fs::read_to_string(path)
        .map_err(|error| vec![diagnostic(None, None, format!("cannot read: {error}"))])?;
    let value: T = toml::from_str(&text).map_err(|error| {
        let line = error.span().map(|span| line_of(&text, span.start));
        vec![diagnostic(line, None, error.message().to_owned())]
    })?;
    let issues = validate(&value);
    if issues.is_empty() {
        return Ok(value);
    }
    Err(issues
        .into_iter()
        .map(|issue| {
            let line = field_line(&text, &issue.field);
            diagnostic(line, Some(issue.field), issue.message)
        })
        .collect())
}

/// 1-based line containing byte `offset`.
fn line_of(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())].matches('\n').count() + 1
}

/// Line of the value at dotted `field` (keys compared case-insensitively, since shader
/// hashes may be written in either case).
fn field_line(text: &str, field: &str) -> Option<usize> {
    let document = DeTable::parse(text).ok()?;
    let mut table = document.get_ref();
    let mut segments = field.split('.').peekable();
    while let Some(segment) = segments.next() {
        let (_, value) = table
            .iter()
            .find(|(key, _)| key.get_ref().eq_ignore_ascii_case(segment))?;
        if segments.peek().is_none() {
            return Some(line_of(text, value.span().start));
        }
        match value.get_ref() {
            DeValue::Table(inner) => table = inner,
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXE: &str = "Game.exe";

    fn setup() -> (tempfile::TempDir, Loader) {
        let dir = tempfile::tempdir().expect("temp dir");
        let loader = Loader::new(dir.path(), EXE);
        (dir, loader)
    }

    /// Writes `text` at `relative` (segments separated by `/`), joined component-wise so
    /// the path renders the way `read_dir` reports it.
    fn write(dir: &Path, relative: &str, text: &str) -> PathBuf {
        let path = relative
            .split('/')
            .fold(dir.to_path_buf(), |path, part| path.join(part));
        fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        fs::write(&path, text).expect("write");
        path
    }

    #[test]
    fn test_empty_dir_is_pass_through_without_diagnostics() {
        let (_dir, mut loader) = setup();
        let snapshot = loader.load();
        assert_eq!(snapshot.settings, None);
        assert!(snapshot.diagnostics.is_empty());
        assert_eq!(snapshot.generation, 1);
    }

    #[test]
    fn test_matching_profile_merges_over_config() {
        let (dir, mut loader) = setup();
        write(
            dir.path(),
            CONFIG_FILE,
            "[stereo]\nmax_parallax_percent = 2.0\n",
        );
        let profile = write(
            dir.path(),
            "profiles/game.toml",
            "executable = \"GAME.EXE\"\n[stereo]\nconvergence = 9.0\n",
        );
        let settings = loader.load().settings.expect("profile matches");
        assert_eq!(settings.stereo.max_parallax_percent, 2.0);
        assert_eq!(settings.stereo.convergence, 9.0);
        assert_eq!(settings.profile, profile);
    }

    #[test]
    fn test_other_executable_stays_pass_through() {
        let (dir, mut loader) = setup();
        write(
            dir.path(),
            "profiles/other.toml",
            "executable = \"other.exe\"\n",
        );
        assert_eq!(loader.load().settings, None);
    }

    #[test]
    fn test_invalid_edit_keeps_last_good_profile_and_reports_line() {
        let (dir, mut loader) = setup();
        let path = write(
            dir.path(),
            "profiles/game.toml",
            "executable = \"Game.exe\"\n[stereo]\nconvergence = 9.0\n",
        );
        assert!(loader.load().settings.is_some());

        fs::write(
            &path,
            "executable = \"Game.exe\"\n[stereo]\nconvergence = -1.0\n",
        )
        .expect("edit");
        let snapshot = loader.load();
        let settings = snapshot.settings.expect("last good profile kept");
        assert_eq!(settings.stereo.convergence, 9.0);
        assert_eq!(
            snapshot.diagnostics,
            [Diagnostic {
                file: path.clone(),
                line: Some(3),
                field: Some("stereo.convergence".to_owned()),
                message: "must be a positive depth, found -1".to_owned(),
            }]
        );
        assert_eq!(
            snapshot.diagnostics[0].to_string(),
            format!(
                "{}:3: stereo.convergence: must be a positive depth, found -1",
                path.display()
            )
        );
    }

    #[test]
    fn test_syntax_error_reports_line_and_keeps_defaults() {
        let (dir, mut loader) = setup();
        write(dir.path(), CONFIG_FILE, "[stereo]\nconvergence = = 3\n");
        write(
            dir.path(),
            "profiles/game.toml",
            "executable = \"Game.exe\"\n",
        );
        let snapshot = loader.load();
        assert_eq!(snapshot.diagnostics.len(), 1);
        assert_eq!(snapshot.diagnostics[0].line, Some(2));
        let settings = snapshot.settings.expect("profile still applies");
        assert_eq!(settings.stereo, StereoParams::default());
    }

    #[test]
    fn test_valid_edit_is_applied_on_next_load() {
        let (dir, mut loader) = setup();
        let path = write(
            dir.path(),
            "profiles/game.toml",
            "executable = \"Game.exe\"\n",
        );
        let first = loader.load();
        fs::write(
            &path,
            "executable = \"Game.exe\"\n[stereo]\nswap_eyes = true\n",
        )
        .expect("edit");
        let second = loader.load();
        assert!(!first.settings.expect("first").stereo.swap_eyes);
        assert!(second.settings.expect("second").stereo.swap_eyes);
        assert!(second.generation > first.generation);
    }

    #[test]
    fn test_duplicate_executable_first_file_wins_with_error() {
        let (dir, mut loader) = setup();
        let first = write(
            dir.path(),
            "profiles/a.toml",
            "executable = \"Game.exe\"\n[stereo]\nconvergence = 5.0\n",
        );
        let second = write(dir.path(), "profiles/b.toml", "executable = \"game.exe\"\n");
        let snapshot = loader.load();
        assert_eq!(snapshot.settings.expect("first wins").profile, first);
        assert_eq!(snapshot.diagnostics.len(), 1);
        assert_eq!(snapshot.diagnostics[0].file, second);
    }

    #[test]
    fn test_deleted_profile_is_forgotten() {
        let (dir, mut loader) = setup();
        let path = write(
            dir.path(),
            "profiles/game.toml",
            "executable = \"Game.exe\"\n",
        );
        assert!(loader.load().settings.is_some());
        fs::remove_file(&path).expect("delete");
        assert_eq!(loader.load().settings, None);
    }

    #[test]
    fn test_field_line_finds_quoted_hash_keys_in_any_case() {
        let text =
            "executable = \"a.exe\"\n\n[shaders]\n\"00000000000000AA\" = { hud_depth = 0.0 }\n";
        assert_eq!(field_line(text, "shaders.00000000000000aa"), Some(4));
        assert_eq!(field_line(text, "stereo.convergence"), None);
    }

    const COMMENTED_PROFILE: &str = "# My game\nexecutable = \"Game.exe\"\n\n[stereo]\n\
        # tuned by hand\nconvergence = 5.0\neye = \"left\"\n\n[shaders]\n\
        \"00000000000000aa\" = \"sky\"\n";

    #[test]
    fn test_save_profile_round_trips_and_keeps_comments_and_entries() {
        let (dir, mut loader) = setup();
        let path = write(dir.path(), "profiles/game.toml", COMMENTED_PROFILE);
        let update = ProfileUpdate {
            max_parallax_percent: Some(3.25),
            convergence: Some(4.2),
            swap_eyes: Some(true),
            matrix: Some(MatrixChoice::Preset(crate::AnaglyphPreset::ZhangLcd)),
        };
        save_profile(&path, &update).expect("save");
        let text = fs::read_to_string(&path).expect("read");
        assert!(
            text.contains("# My game") && text.contains("# tuned by hand"),
            "{text}"
        );
        assert!(text.contains("convergence = 4.2\n"), "{text}");
        assert!(text.contains("matrix = \"zhang-lcd\""), "{text}");

        let settings = loader.load().settings.expect("profile still matches");
        assert_eq!(settings.stereo.max_parallax_percent, 3.25);
        assert_eq!(settings.stereo.convergence, 4.2);
        assert!(settings.stereo.swap_eyes);
        assert_eq!(settings.stereo.eye, Some(crate::Eye::Left));
        assert_eq!(update.matrix, Some(settings.anaglyph.matrix));
        assert_eq!(settings.shaders.len(), 1);
    }

    #[test]
    fn test_save_profile_writes_custom_matrix_and_skips_unset_fields() {
        let (dir, mut loader) = setup();
        let path = write(dir.path(), "profiles/game.toml", COMMENTED_PROFILE);
        let matrix = crate::AnaglyphPreset::Gray.matrix();
        let update = ProfileUpdate {
            matrix: Some(MatrixChoice::Custom(matrix)),
            ..ProfileUpdate::default()
        };
        save_profile(&path, &update).expect("save");
        let settings = loader.load().settings.expect("profile still matches");
        assert_eq!(settings.anaglyph.matrix, MatrixChoice::Custom(matrix));
        assert_eq!(settings.stereo.convergence, 5.0);
        assert_eq!(
            settings.stereo.max_parallax_percent,
            StereoParams::default().max_parallax_percent
        );
    }

    #[test]
    fn test_save_profile_refuses_invalid_values_and_leaves_file_unchanged() {
        let (dir, _) = setup();
        let path = write(dir.path(), "profiles/game.toml", COMMENTED_PROFILE);
        let update = ProfileUpdate {
            convergence: Some(-1.0),
            ..ProfileUpdate::default()
        };
        let error = save_profile(&path, &update).expect_err("negative convergence");
        assert!(matches!(error, Error::ProfileEdit { .. }), "{error}");
        assert_eq!(fs::read_to_string(&path).expect("read"), COMMENTED_PROFILE);
    }
}

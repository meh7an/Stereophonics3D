//! Panic containment for functions exported across the C ABI.

use std::{
    any::Any,
    panic::{self, AssertUnwindSafe},
};

/// Runs `body` and returns its result, or `fallback` if `body` panics.
///
/// Every function exported across the C ABI wraps its body in this guard: a panic that
/// reaches an `extern "C"` boundary aborts the host process, and the host is a game.
/// The panic is logged with `entry_point` so the failing hook can be identified.
///
/// Requires the default `panic = "unwind"` strategy; under `panic = "abort"` nothing
/// can be caught.
pub fn ffi_guard<T>(entry_point: &'static str, fallback: T, body: impl FnOnce() -> T) -> T {
    // Unwind safety cannot be proven for arbitrary hook bodies. State shared between
    // hooks lives behind locks, which poison on panic instead of exposing torn data.
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            tracing::error!(
                entry_point,
                panic = panic_message(payload.as_ref()),
                "panic caught at FFI boundary"
            );
            fallback
        }
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else {
        "non-string panic payload"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_guard_normal_body_returns_value() {
        assert_eq!(ffi_guard("test", -1, || 7), 7);
    }

    #[test]
    fn test_ffi_guard_panicking_body_returns_fallback() {
        let result = ffi_guard("test", -1, || -> i32 { panic!("boom") });
        assert_eq!(result, -1);
    }

    #[test]
    fn test_panic_message_formatted_payload_returns_text() {
        let payload = panic::catch_unwind(|| panic!("code {}", 42)).unwrap_err();
        assert_eq!(panic_message(payload.as_ref()), "code 42");
    }

    #[test]
    fn test_panic_message_static_payload_returns_text() {
        let payload = panic::catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(payload.as_ref()), "static");
    }
}

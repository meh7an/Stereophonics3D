//! Function-pointer plumbing shared by every hook module.

/// Casts a function returned by a `GetProcAddr` to its real signature; only used inside
/// unsafe blocks, where the loader guarantees the named entry point has signature `$ty`.
macro_rules! cast {
    ($function:expr, $ty:ty) => {
        $function.map(|f| std::mem::transmute::<unsafe extern "system" fn(), $ty>(f))
    };
}

/// Exposes one of this layer's hooks as a `PFN_vkVoidFunction`.
macro_rules! expose {
    ($function:expr, $ty:ty) => {
        // SAFETY: function pointers of every signature have the same representation; the
        // loader casts it back to `$ty` before calling.
        Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>($function) })
    };
}

/// Declares a table of next-layer functions, loaded by name, together with the hook
/// that replaces each one. `hook` hands out a replacement only when the next layer
/// implements the function, so a hook never runs without somewhere to forward to.
macro_rules! function_table {
    (
        $(#[$meta:meta])*
        $table:ident, $handle:ty, $get_proc_addr:ty {
            $($field:ident: $name:literal => $ty:ty = $hook:path,)*
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy)]
        pub struct $table {
            $(pub $field: Option<$ty>,)*
        }

        impl $table {
            /// Resolves every function through the next layer's `GetProcAddr`.
            ///
            /// # Safety
            ///
            /// `get_proc_addr` must be the next layer's function and `handle` a live
            /// handle it accepts.
            pub unsafe fn load(get_proc_addr: $get_proc_addr, handle: $handle) -> Self {
                // SAFETY: each name is a nul-terminated entry point of signature `$ty`.
                unsafe {
                    Self {
                        $($field: cast!(get_proc_addr(handle, $name.as_ptr()), $ty),)*
                    }
                }
            }

            /// This layer's replacement for entry point `name`, if it hooks it and the
            /// next layer provides it.
            pub fn hook(&self, name: &std::ffi::CStr) -> Option<ash::vk::PFN_vkVoidFunction> {
                $(
                    if self.$field.is_some() && name == $name {
                        return Some(expose!($hook, $ty));
                    }
                )*
                None
            }
        }
    };
}

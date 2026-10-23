# Third-Party Code and Licences

Stereophonics3D is licensed under the GNU General Public License v3.0 (see `LICENSE`).
This file is the single register of code taken from other projects. Every import is
recorded here in the same change that adds it.

## Reused Source Code

| Source | Licence | What was taken | Where it lives |
|---|---|---|---|
| [font8x8](https://github.com/dhepper/font8x8) `font8x8_basic.h` by Daniel Hepper, from Marcel Sondaar's IBM VGA-based font | Public domain | 8x8 bitmaps of U+0020 to U+007E (readout font) | `layer/src/readout.rs` (`FONT`) |

## Statically Linked Libraries

| Library | Licence | Source | Linked into |
|---|---|---|---|
| SPIRV-Tools (`SPIRV-Tools.lib`, validator C API) | Apache-2.0 | LunarG Vulkan SDK 1.4.363.0 | `s3d-spirv-val` and every binary that depends on it |

## Crate Dependencies

Crates pulled from crates.io keep their own licences. The full list is generated from
`Cargo.lock` before each release.

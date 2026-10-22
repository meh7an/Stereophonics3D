#!/usr/bin/env bash
# Regenerates fixtures/spirv/*.spv from src/ with the Vulkan SDK's compilers.
# The .spv files are committed so tests never depend on the installed compiler version;
# run this only when a source changes or the corpus grows, and commit both together.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sdk="${VULKAN_SDK:?VULKAN_SDK is not set}"
if command -v cygpath >/dev/null; then
    sdk="$(cygpath -u "$sdk")"
fi
bin="$sdk/Bin"
src="$here/src"

glsl() {
    local env="$1" name="$2" source="$3"
    shift 3
    "$bin/glslc" --target-env="$env" "$@" "$src/$source" -o "$here/$name.spv"
}

hlsl() {
    local env="$1" name="$2"
    "$bin/dxc" -spirv -fspv-target-env="$env" -T vs_6_0 -E main "$src/$name.hlsl" \
        -Fo "$here/$name.spv"
}

glsl vulkan1.0 glsl_basic glsl_basic.vert -O
glsl vulkan1.3 glsl_piecewise glsl_piecewise.vert
glsl vulkan1.0 glsl_redeclared glsl_redeclared.vert -O
glsl vulkan1.0 glsl_function glsl_function.vert
glsl vulkan1.0 glsl_tese glsl_tese.tese -O
glsl vulkan1.0 glsl_geom glsl_geom.geom -O
glsl vulkan1.3 glsl_mesh glsl_mesh.mesh -O
glsl vulkan1.0 glsl_frag glsl_frag.frag -O
hlsl vulkan1.0 hlsl_basic
hlsl vulkan1.3 hlsl_struct
"$bin/spirv-as" --target-env vulkan1.0 "$src/multi_entry.spvasm" -o "$here/multi_entry.spv"

echo "fixtures regenerated in $here"

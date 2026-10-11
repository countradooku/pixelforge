# Vendored GPUI (gpui-pre 0.3.8)

These four crates are unmodified copies of the crates.io releases, apart from the
changes listed below. `Cargo.toml` points at them with `[patch.crates-io]`:

- `gpui-pre`
- `gpui-pre-apple` (Metal)
- `gpui-pre-wgpu` (WGSL: Vulkan, GL, WebGPU and WebGL)
- `gpui-pre-windows` (DirectX, HLSL)

## Why

Rotate View draws the cached display tiles as rotated sprites on the GPU. Stock GPUI can only
draw axis-aligned images.

## Changes

- `PolychromeSprite` gains a `transformation: TransformationMatrix` (device pixels, row-major
  2×2 plus translation), appended so the struct stays 8-byte aligned (96 → 120 bytes).
  `gpui-pre-apple/vendor/gpui/src/scene.rs` is the copy cbindgen reads for the Metal header;
  it has the same change.
- `Window::paint_image_transformed(bounds, image_bounds, image, frame, transformation)`. Its
  bounds are not snapped to device pixels, so adjacent tiles share exact edges.
  `paint_image` is unchanged and passes the unit matrix.
- `Scene::insert_primitive` culls and orders a transformed sprite by the bounding box of its
  transformed quad.
- Polychrome sprite shaders, in `shaders.metal`, `shaders.wgsl` + `shaders_webgl.wgsl` and
  `shaders.hlsl`:
  - The vertex stage transforms the quad and its clip distances, the same way monochrome
    sprites already did.
  - The corner-radius SDF is evaluated in the sprite's own (untransformed) space.
  - For transformed sprites only, the fragment stage clamps texture lookups to the tile's
    texels and skips edge antialiasing. The caller overlaps neighbouring tiles by half a
    device pixel, so no seams show.
- `wgpu_renderer.rs`: the size assertion is 30 words, and the WebGL instance stride is 30.

## Verification

- Metal is compiled by `gpui-pre-apple`'s build script on every macOS build.
- WGSL (storage-buffer and WebGL variants) passed Chromium's Tint validator through WebGPU,
  just like the originals.
- HLSL mirrors the monochrome-sprite code it reuses, but hasn't been compiled: there is no
  DirectX compiler on macOS.

## block 0.1.6 compatibility fix

`block` is the existing MIT-licensed transitive dependency, copied from the
local Cargo cache without a download or version change. Its foreign
`_NSConcreteStackBlock` declaration now uses an inhabited byte type and casts
the symbol address to the existing opaque class pointer. No symbol contents
are read. This removes Rust's `uninhabited_static` future-compatibility warning
without changing the block layout or runtime ABI. Foreign functions also name
their existing C ABI explicitly to satisfy the current compiler.

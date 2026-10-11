pub mod actions;
pub mod caf;
pub mod dock;
pub mod docs;
pub mod editor;
mod render;
pub mod tools;
pub mod type_tools;
pub mod vector_tools;
pub mod web_ui;
pub mod viewport;
pub mod vp;
pub mod perspwarp_ui;
mod demo;
mod demo_gaps;
mod demo_gpu;
pub mod layers_ui;
pub mod nav_tools;
pub mod options;
pub mod paint_tools;
pub mod retouch_tools;
pub mod select_mask;
pub mod select_tools;
pub mod shape_props;
pub mod style_ui;

/// RGBA8 previews, thumbnails and overlays share the document memory allowance.
/// Reserve CPU pixels and the matching atlas upload; GPUI may retain an Arc
/// after a panel closes, so the charge follows that Arc rather than the panel.
pub fn render_image(frames: Vec<image::Frame>) -> std::sync::Arc<gpui_kit::RenderImage> {
    let bytes = frames.iter().map(|f| u64::from(f.buffer().width()) * u64::from(f.buffer().height()) * 8).sum();
    crate::engine::memory::global().track_shared(std::sync::Arc::new(gpui_kit::RenderImage::new(frames)), bytes)
}

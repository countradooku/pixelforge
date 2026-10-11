//! PixelForge: a tiled, mipmapped, multi-threaded raster editor on GPUI.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod bench;
mod bench_memory;
mod bench_config;
mod bench_resources;
mod bench_gpu;
mod compare;
mod compare_gpu;
mod engine;
mod gtmask;
mod psd_cli;
mod ui;

use gpui_kit::component::{Theme, ThemeMode};
use gpui_kit::{AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px, size};

use ui::actions;
use ui::editor::Editor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match psd_cli::run(&args) {
        Ok(true) => return,
        Ok(false) => {},
        Err(error) => { eprintln!("{error:#}"); std::process::exit(1); }
    }
    if let Some((i, suite)) = args.iter().enumerate().find_map(|(i, a)| bench_config::Suite::parse(a).map(|s| (i, s))) {
        let options = match bench_config::Options::parse(suite, &args[i + 1..]) {
            Ok(options) => options,
            Err(error) => {
                eprintln!("{error}\n\n{}", bench_config::HELP);
                std::process::exit(2);
            }
        };
        if options.list {
            bench_config::print_tiers();
            return;
        }
        if suite == bench_config::Suite::Gpu {
            bench_gpu::run(&options.sizes, options.only.as_deref(), options.write_priors);
        } else {
            for d in options.sizes {
                let _resources = bench_resources::Report::start();
                match suite {
                    bench_config::Suite::Engine => bench::run(d.w, d.h),
                    bench_config::Suite::Filters => bench::run_filters(d.w, d.h, options.only.as_deref()),
                    bench_config::Suite::Memory => bench_memory::run(d.w, d.h, options.only.as_deref()),
                    bench_config::Suite::Tools => bench::run_tools(d.w, d.h, options.only.as_deref()),
                    bench_config::Suite::Gpu => unreachable!(),
                }
            }
        }
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--export-bench-tier") {
        let tier = args.get(i + 1).and_then(|v| bench_config::TIERS.iter().find(|(name, _)| name == v));
        let (Some((_, d)), Some(path)) = (tier, args.get(i + 2)) else { eprintln!("usage: --export-bench-tier 24|61|150|268 path.tif"); std::process::exit(2); };
        let layer = engine::document::Layer::new(1, "Benchmark", engine::ops::procedural(d.w, d.h));
        engine::io::export(&[layer], d.w, d.h, std::path::Path::new(path)).expect("export benchmark fixture");
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--export-scene") {
        let side: u32 = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(16384);
        let path = std::path::PathBuf::from(args.get(i + 2).cloned().unwrap_or_else(|| "scene.tif".into()));
        let g = engine::ops::demo_scene(side, side);
        let layer = engine::document::Layer::new(1, "scene", g);
        match engine::io::export(&[layer], side, side, &path) {
            Ok(()) => println!("wrote {}", path.display()),
            Err(e) => eprintln!("error: {e:#}"),
        }
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--compare-gaps") {
        compare::run(&args[i + 1..]);
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--gt-mask") {
        if let Err(e) = gtmask::run(&args[i + 1..]) {
            eprintln!("error: {e:#}");
        }
        return;
    }
    // The GPU context is created in the background; until it's ready (and
    // whenever it isn't usable) everything runs on the CPU.
    engine::prefs::Prefs::load().apply_gpu();
    engine::gpu::set_blocking(false);
    engine::gpu::warm_up();
    gpui_kit::application().with_assets(gpui_kit::assets::AllAssets).run(|cx| {
        gpui_kit::init(cx);
        Theme::change(ThemeMode::Dark, None, cx);
        actions::bind_keys(cx);
        actions::set_menus(cx);
        cx.on_action(|_: &actions::Quit, cx| cx.quit());

        let bounds = Bounds::centered(None, size(px(1680.), px(1050.)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions { title: Some("PixelForge".into()), ..Default::default() }),
            ..Default::default()
        };
        let (_, editor) = gpui_kit::open_window(options, cx, |window, cx| cx.new(|cx| Editor::new(window, cx)))
            .expect("failed to open window");
        let _ = editor;
        cx.activate(true);
    });
}

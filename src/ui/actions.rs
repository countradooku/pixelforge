//! Menu bar, actions and keyboard shortcuts.

use gpui_kit::{App, KeyBinding, Menu, MenuItem};

gpui_kit::actions!(
    pixelforge,
    [
        Quit,
        Preferences,
        NewDocument,
        CloseDocument,
        ShowBrushes,
        ShowBrushSettings,
        ShowMeasurementLog,
        ShowNotes,
        ShowChannels,
        RecordMeasurements,
        NextDocument,
        PrevDocument,
        ArrangeTabs,
        ArrangeTwoUp,
        ArrangeTile,
        MatchZoom,
        MatchLocation,
        MatchRotation,
        MatchAll,
        NewBenchmark,
        Open,
        Save,
        SaveAs,
        EditUndo,
        EditRedo,
        EditCut,
        EditCopy,
        EditPaste,
        FillForeground,
        FillBackground,
        ClearPixels,
        SelectAll,
        SelectAllLayers,
        Deselect,
        InvertSelection,
        ZoomIn,
        ZoomOut,
        ZoomFit,
        ZoomActual,
        NewLayer,
        DuplicateLayer,
        DeleteLayer,
        MergeDown,
        FlattenImage,
        LayerUp,
        LayerDown,
        LayerViaCopy,
        LayerViaCut,
        FlipHorizontal,
        FlipVertical,
        RotateCw,
        RotateCcw,
        CropToSelection,
        RevealAll,
        ImageSize,
        SelectSubject,
        SelectAndMask,
        ContentAwareFill,
        PerspectiveWarp,
        QuickMaskMode,
        ScreenMode,
        GroupLayers,
        UngroupLayers,
        ClippingMask,
        PlaceEmbedded,
        RasterizeLayer,
        ConvertToFrame,
        MergeShapeComponents,
        GroupIsolation,
    ]
);

/// Layer ▸ Layer Mask ▸ … ("add", "delete", "apply", "invert", "toggle").
#[derive(Clone, PartialEq, Debug, gpui_kit::Action)]
#[action(namespace = pixelforge, no_json)]
pub struct LayerMaskOp(pub &'static str);

/// Layer ▸ Layer Style ▸ … (`style_ui::MENU` keys, or "clear").
#[derive(Clone, PartialEq, Debug, gpui_kit::Action)]
#[action(namespace = pixelforge, no_json)]
pub struct LayerStyleOp(pub &'static str);

/// Layer ▸ New Adjustment Layer ▸ …
#[derive(Clone, PartialEq, Debug, gpui_kit::Action)]
#[action(namespace = pixelforge, no_json)]
pub struct NewAdjustment(pub Kind);

/// Select ▸ Modify ▸ …
#[derive(Clone, PartialEq, Debug, gpui_kit::Action)]
#[action(namespace = pixelforge, no_json)]
pub struct ModifySelection(pub super::select_tools::Modify);

use crate::engine::filters::catalog::{Kind, Menu as FilterMenu};

/// Runs (or opens the dialog for) a catalog filter.
#[derive(Clone, PartialEq, Debug, gpui_kit::Action)]
#[action(namespace = pixelforge, no_json)]
pub struct RunFilter(pub Kind);

fn item(k: Kind) -> MenuItem {
    let e = k.entry();
    let label = if e.instant() || k == Kind::Liquify { e.title.to_string() } else { format!("{}…", e.title) };
    MenuItem::action(label, RunFilter(k))
}

fn filter_items(menu: FilterMenu) -> Vec<MenuItem> {
    Kind::all().into_iter().filter(|k| k.entry().menu == menu).map(item).collect()
}

/// Filter ▸ Filter Gallery ▸ {Artistic, Brush Strokes, …}.
fn gallery_menu() -> MenuItem {
    use crate::engine::filters::gallery::G;
    let groups = G::GROUPS.iter().map(|&grp| {
        let items: Vec<MenuItem> = G::ALL.iter().filter(|g| g.group() == grp).map(|&g| item(Kind::Gallery(g))).collect();
        MenuItem::submenu(Menu::new(grp).items(items))
    });
    MenuItem::submenu(Menu::new("Filter Gallery").items(groups.collect::<Vec<_>>()))
}

fn filter_menu() -> Menu {
    let mut items = vec![gallery_menu(), MenuItem::submenu(Menu::new("Neural Filters").items(filter_items(FilterMenu::Neural)))];
    items.extend(filter_items(FilterMenu::Top));
    items.push(MenuItem::separator());
    items.extend(FilterMenu::FILTER_MENUS.iter().map(|&m| MenuItem::submenu(Menu::new(m.title()).items(filter_items(m)))));
    Menu::new("Filter").items(items)
}

pub const CONTEXT: &str = "PixelForge";

pub fn bind_keys(cx: &mut App) {
    let c = Some(CONTEXT);
    cx.bind_keys([
        KeyBinding::new("cmd-q", Quit, None),
        KeyBinding::new("cmd-,", Preferences, c),
        KeyBinding::new("cmd-n", NewDocument, c),
        KeyBinding::new("cmd-o", Open, c),
        KeyBinding::new("cmd-w", CloseDocument, c),
        KeyBinding::new("f5", ShowBrushSettings, c),
        KeyBinding::new("cmd-shift-m", RecordMeasurements, c),
        KeyBinding::new("ctrl-tab", NextDocument, c),
        KeyBinding::new("ctrl-shift-tab", PrevDocument, c),
        KeyBinding::new("cmd-s", Save, c),
        KeyBinding::new("cmd-shift-s", SaveAs, c),
        KeyBinding::new("cmd-z", EditUndo, c),
        KeyBinding::new("cmd-shift-z", EditRedo, c),
        KeyBinding::new("cmd-x", EditCut, c),
        KeyBinding::new("cmd-c", EditCopy, c),
        KeyBinding::new("cmd-v", EditPaste, c),
        KeyBinding::new("alt-backspace", FillForeground, c),
        KeyBinding::new("cmd-backspace", FillBackground, c),
        KeyBinding::new("backspace", ClearPixels, c),
        KeyBinding::new("delete", ClearPixels, c),
        KeyBinding::new("cmd-a", SelectAll, c),
        KeyBinding::new("alt-cmd-a", SelectAllLayers, c),
        KeyBinding::new("cmd-d", Deselect, c),
        KeyBinding::new("cmd-shift-i", InvertSelection, c),
        KeyBinding::new("cmd-=", ZoomIn, c),
        KeyBinding::new("cmd-+", ZoomIn, c),
        KeyBinding::new("cmd--", ZoomOut, c),
        KeyBinding::new("cmd-0", ZoomFit, c),
        KeyBinding::new("cmd-1", ZoomActual, c),
        KeyBinding::new("cmd-shift-n", NewLayer, c),
        KeyBinding::new("cmd-j", LayerViaCopy, c),
        KeyBinding::new("cmd-shift-j", LayerViaCut, c),
        KeyBinding::new("cmd-e", MergeDown, c),
        KeyBinding::new("cmd-shift-e", FlattenImage, c),
        KeyBinding::new("cmd-]", LayerUp, c),
        KeyBinding::new("cmd-[", LayerDown, c),
        KeyBinding::new("cmd-i", RunFilter(Kind::Invert), c),
        KeyBinding::new("cmd-shift-u", RunFilter(Kind::Desaturate), c),
        KeyBinding::new("cmd-u", RunFilter(Kind::HueSaturation), c),
        KeyBinding::new("cmd-l", RunFilter(Kind::Levels), c),
        KeyBinding::new("cmd-m", RunFilter(Kind::Curves), c),
        KeyBinding::new("cmd-b", RunFilter(Kind::ColorBalance), c),
        KeyBinding::new("cmd-shift-l", RunFilter(Kind::AutoTone), c),
        KeyBinding::new("cmd-alt-shift-l", RunFilter(Kind::AutoContrast), c),
        KeyBinding::new("cmd-alt-shift-b", RunFilter(Kind::BlackWhite), c),
        KeyBinding::new("cmd-shift-x", RunFilter(Kind::Liquify), c),
        KeyBinding::new("cmd-t", RunFilter(Kind::FreeTransform), c),
        KeyBinding::new("cmd-alt-v", RunFilter(Kind::VanishingPoint), c),
        KeyBinding::new("cmd-alt-shift-a", RunFilter(Kind::AdaptiveWideAngle), c),
        KeyBinding::new("cmd-shift-a", RunFilter(Kind::CameraRaw), c),
        KeyBinding::new("cmd-shift-r", RunFilter(Kind::LensCorrection), c),
        KeyBinding::new("cmd-alt-i", ImageSize, c),
        KeyBinding::new("cmd-g", GroupLayers, c),
        KeyBinding::new("cmd-shift-g", UngroupLayers, c),
        KeyBinding::new("cmd-alt-g", ClippingMask, c),
        KeyBinding::new("shift-f6", ModifySelection(super::select_tools::Modify::Feather), c),
        KeyBinding::new("cmd-alt-d", ModifySelection(super::select_tools::Modify::Feather), c),
        KeyBinding::new("cmd-alt-r", SelectAndMask, c),
    ]);
}

pub fn set_menus(cx: &mut App) {
    cx.set_menus([
        Menu::new("PixelForge").items([MenuItem::action("Settings… (Performance)", Preferences), MenuItem::separator(), MenuItem::action("Quit PixelForge", Quit)]),
        Menu::new("File").items([
            MenuItem::action("New…", NewDocument),
            MenuItem::action("New 256 MP Test Image", NewBenchmark),
            MenuItem::action("Open…", Open),
            MenuItem::action("Close", CloseDocument),
            MenuItem::action("Place Embedded…", PlaceEmbedded),
            MenuItem::separator(),
            MenuItem::action("Save", Save),
            MenuItem::action("Save As / Export…", SaveAs),
        ]),
        Menu::new("Edit").items([
            MenuItem::action("Undo", EditUndo),
            MenuItem::action("Redo", EditRedo),
            MenuItem::separator(),
            MenuItem::action("Cut", EditCut),
            MenuItem::action("Copy", EditCopy),
            MenuItem::action("Paste as Layer", EditPaste),
            MenuItem::separator(),
            MenuItem::submenu(Menu::new("Transform").items(filter_items(FilterMenu::Transform))),
            MenuItem::action("Perspective Warp", PerspectiveWarp),
            MenuItem::separator(),
            MenuItem::action("Fill with Foreground", FillForeground),
            MenuItem::action("Fill with Background", FillBackground),
            MenuItem::action("Content-Aware Fill…", ContentAwareFill),
            MenuItem::action("Clear", ClearPixels),
        ]),
        Menu::new("Image").items([
            MenuItem::submenu(Menu::new("Adjustments").items(filter_items(FilterMenu::Adjustments))),
            MenuItem::separator(),
            MenuItem::action("Auto Tone", RunFilter(Kind::AutoTone)),
            MenuItem::action("Auto Contrast", RunFilter(Kind::AutoContrast)),
            MenuItem::separator(),
            MenuItem::action("Image Size…", ImageSize),
            MenuItem::action("Crop to Selection", CropToSelection),
            MenuItem::action("Reveal All", RevealAll),
            MenuItem::separator(),
            MenuItem::submenu(Menu::new("Analysis").items([
                MenuItem::action("Record Measurements", RecordMeasurements),
                MenuItem::action("Measurement Log (Set Scale)…", ShowMeasurementLog),
            ])),
            MenuItem::separator(),
            MenuItem::action("Rotate 90° Clockwise", RotateCw),
            MenuItem::action("Rotate 90° Counter-Clockwise", RotateCcw),
            MenuItem::action("Flip Canvas Horizontal", FlipHorizontal),
            MenuItem::action("Flip Canvas Vertical", FlipVertical),
        ]),
        Menu::new("Layer").items([
            MenuItem::action("New Layer", NewLayer),
            MenuItem::action("Duplicate Layer", DuplicateLayer),
            MenuItem::action("Delete Layer", DeleteLayer),
            MenuItem::separator(),
            MenuItem::submenu(Menu::new("New Adjustment Layer").items(
                super::layers_ui::adjustment_kinds().into_iter().map(|k| MenuItem::action(format!("{}…", k.entry().title), NewAdjustment(k))).collect::<Vec<_>>(),
            )),
            MenuItem::submenu(Menu::new("Layer Mask").items([
                MenuItem::action("Add (Reveal Selection / All)", LayerMaskOp("add")),
                MenuItem::action("Invert", LayerMaskOp("invert")),
                MenuItem::action("Enable / Disable", LayerMaskOp("toggle")),
                MenuItem::action("Apply", LayerMaskOp("apply")),
                MenuItem::action("Delete", LayerMaskOp("delete")),
            ])),
            MenuItem::action("Create / Release Clipping Mask", ClippingMask),
            MenuItem::submenu(Menu::new("Layer Style").items({
                let mut items: Vec<MenuItem> = super::style_ui::MENU.iter().map(|&(k, label)| MenuItem::action(label, LayerStyleOp(k))).collect();
                items.insert(1, MenuItem::separator());
                items.push(MenuItem::separator());
                items.push(MenuItem::action("Clear Layer Style", LayerStyleOp("clear")));
                items
            })),
            MenuItem::separator(),
            MenuItem::action("Group Layers", GroupLayers),
            MenuItem::action("Ungroup Layers", UngroupLayers),
            MenuItem::action("Group: Pass Through / Isolated", GroupIsolation),
            MenuItem::action("Rasterize Layer", RasterizeLayer),
            MenuItem::action("Convert to Frame", ConvertToFrame),
            MenuItem::action("Merge Shape Components", MergeShapeComponents),
            MenuItem::separator(),
            MenuItem::action("Layer via Copy", LayerViaCopy),
            MenuItem::action("Layer via Cut", LayerViaCut),
            MenuItem::separator(),
            MenuItem::action("Bring Forward", LayerUp),
            MenuItem::action("Send Backward", LayerDown),
            MenuItem::separator(),
            MenuItem::action("Merge Down", MergeDown),
            MenuItem::action("Flatten Image", FlattenImage),
        ]),
        Menu::new("Select").items([
            MenuItem::action("All", SelectAll),
            MenuItem::action("Deselect", Deselect),
            MenuItem::action("All Layers", SelectAllLayers),
            MenuItem::action("Inverse", InvertSelection),
            MenuItem::separator(),
            MenuItem::action("Subject", SelectSubject),
            MenuItem::action("Select and Mask…", SelectAndMask),
            MenuItem::submenu(Menu::new("Modify").items([
                MenuItem::action("Border…", ModifySelection(super::select_tools::Modify::Border)),
                MenuItem::action("Smooth…", ModifySelection(super::select_tools::Modify::Smooth)),
                MenuItem::action("Expand…", ModifySelection(super::select_tools::Modify::Expand)),
                MenuItem::action("Contract…", ModifySelection(super::select_tools::Modify::Contract)),
                MenuItem::action("Feather…", ModifySelection(super::select_tools::Modify::Feather)),
            ])),
            MenuItem::separator(),
            MenuItem::action("Edit in Quick Mask Mode", QuickMaskMode),
        ]),
        filter_menu(),
        Menu::new("View").items([
            MenuItem::action("Zoom In", ZoomIn),
            MenuItem::action("Zoom Out", ZoomOut),
            MenuItem::action("Fit on Screen", ZoomFit),
            MenuItem::action("100%", ZoomActual),
            MenuItem::separator(),
            MenuItem::action("Screen Mode", ScreenMode),
        ]),
        Menu::new("Window").items([
            MenuItem::submenu(Menu::new("Arrange").items([
                MenuItem::action("Consolidate All to Tabs", ArrangeTabs),
                MenuItem::action("2-up Vertical", ArrangeTwoUp),
                MenuItem::action("Tile All in Grid", ArrangeTile),
                MenuItem::separator(),
                MenuItem::action("Match Zoom", MatchZoom),
                MenuItem::action("Match Location", MatchLocation),
                MenuItem::action("Match Rotation", MatchRotation),
                MenuItem::action("Match All", MatchAll),
            ])),
            MenuItem::separator(),
            MenuItem::action("Brushes", ShowBrushes),
            MenuItem::action("Brush Settings", ShowBrushSettings),
            MenuItem::action("Measurement Log", ShowMeasurementLog),
            MenuItem::action("Notes", ShowNotes),
            MenuItem::action("Channels", ShowChannels),
            MenuItem::separator(),
            MenuItem::action("Next Document", NextDocument),
            MenuItem::action("Previous Document", PrevDocument),
            MenuItem::action("Close Document", CloseDocument),
        ]),
    ]);
}

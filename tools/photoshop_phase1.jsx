// Phase 1: execute against the exact opaque RGBA8 benchmark fixture exported by
// PixelForge. Run only through photoshop_phase1.py, which watches CG windows.
#target photoshop
var output = [];
var priorDialogs = app.displayDialogs;
var priorDoc = app.documents.length ? app.activeDocument : null;
var doc = null;
function timed(name, fn) {
    var best = 1e30;
    for (var n = 0; n < 3; n++) {
        doc.activeHistoryState = base;
        $.hiresTimer; fn(); var ms = $.hiresTimer / 1000;
        best = Math.min(best, ms);
    }
    output.push(name + "\t" + best.toFixed(3));
}
try {
    app.displayDialogs = DialogModes.NO;
    $.hiresTimer; doc = app.open(new File(INPUT));
    output.push("Fixture open\t" + ($.hiresTimer / 1000).toFixed(3));
    var base = doc.activeHistoryState;
    timed("Hue/Saturation", function () {
        var d = new ActionDescriptor(), list = new ActionList(), values = new ActionDescriptor();
        values.putInteger(charIDToTypeID("H   "), 40);
        values.putInteger(charIDToTypeID("Strt"), 20);
        values.putInteger(charIDToTypeID("Lght"), 0);
        list.putObject(charIDToTypeID("Hst2"), values);
        d.putList(charIDToTypeID("Adjs"), list);
        d.putBoolean(charIDToTypeID("Clrz"), false);
        executeAction(charIDToTypeID("HStr"), d, DialogModes.NO);
    });
    timed("Levels", function () { doc.activeLayer.adjustLevels(20, 230, 1.2, 0, 255); });
    timed("Gaussian blur r20", function () { doc.activeLayer.applyGaussianBlur(20); });
    timed("Gaussian blur r120", function () { doc.activeLayer.applyGaussianBlur(120); });
    timed("Invert", function () { doc.activeLayer.invert(); });
    doc.activeHistoryState = base;
    var saveOptions = new PhotoshopSaveOptions(); saveOptions.layers = true;
    $.hiresTimer; doc.saveAs(new File(OUTPUT), saveOptions, true);
    output.push("PSD save\t" + ($.hiresTimer / 1000).toFixed(3));
} catch (e) { output.push("ERROR\t" + e.message); }
finally {
    if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
    if (priorDoc) app.activeDocument = priorDoc;
    app.displayDialogs = priorDialogs;
}
output.join("\n");

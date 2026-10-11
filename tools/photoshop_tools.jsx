// Times Photoshop's scriptable equivalents of PixelForge tool operations on
// the same image (`pixelforge --export-scene 16384 scene.tif`).
// Run through tools/photoshop_tools.sh, which sets SCENE and PHASES.
#target photoshop
app.displayDialogs = DialogModes.NO;
var out = [];
function cTID(s) { return charIDToTypeID(s); }
function sTID(s) { return stringIDToTypeID(s); }
var doc = app.open(new File(SCENE));
var S = doc.width.as("px");
var base = doc.historyStates.length - 1;
function reset() { doc.activeHistoryState = doc.historyStates[base]; }

// Best of two runs; `prep` runs before the timer starts.
function t(name, prep, fn) {
  try {
    var best = 1e15;
    for (var k = 0; k < 2; k++) {
      reset();
      if (prep) prep();
      $.hiresTimer;
      fn();
      var us = $.hiresTimer;
      if (us < best) best = us;
    }
    out.push(name + "\t" + (best / 1000).toFixed(1));
  } catch (e) { out.push(name + "\tERR " + String(e.message).split("\n")[0]); }
}

function rectSel(x0, y0, x1, y1, type) {
  doc.selection.select([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], type || SelectionType.REPLACE, 0, false);
}
function ellipseSel(x0, y0, x1, y1, add) {
  var d = new ActionDescriptor();
  var r = new ActionReference(); r.putProperty(cTID("Chnl"), cTID("fsel"));
  d.putReference(cTID("null"), r);
  var e = new ActionDescriptor();
  e.putUnitDouble(cTID("Top "), cTID("#Pxl"), y0);
  e.putUnitDouble(cTID("Left"), cTID("#Pxl"), x0);
  e.putUnitDouble(cTID("Btom"), cTID("#Pxl"), y1);
  e.putUnitDouble(cTID("Rght"), cTID("#Pxl"), x1);
  d.putObject(cTID("T   "), cTID("Elps"), e);
  d.putBoolean(cTID("AntA"), true);
  executeAction(add ? cTID("AddT") : cTID("setd"), d, DialogModes.NO);
}
function wand(x, y, tol, contiguous, allLayers) {
  var d = new ActionDescriptor();
  var r = new ActionReference(); r.putProperty(cTID("Chnl"), cTID("fsel"));
  d.putReference(cTID("null"), r);
  var p = new ActionDescriptor();
  p.putUnitDouble(cTID("Hrzn"), cTID("#Pxl"), x);
  p.putUnitDouble(cTID("Vrtc"), cTID("#Pxl"), y);
  d.putObject(cTID("T   "), cTID("Pnt "), p);
  d.putInteger(cTID("Tlrn"), tol);
  d.putBoolean(cTID("AntA"), true);
  if (!contiguous) d.putBoolean(cTID("Cntg"), false);
  if (allLayers) d.putBoolean(cTID("Mrgd"), true);
  executeAction(cTID("setd"), d, DialogModes.NO);
}
function selectionUnion() {
  rectSel(S / 10, S / 10, S * 6 / 10, S * 6 / 10);
  ellipseSel(S * 3 / 10, S * 3 / 10, S * 9 / 10, S * 9 / 10, true);
}

if (PHASES.indexOf("1") >= 0) {
  t("selection add (rect + ellipse)", function () { rectSel(S / 10, S / 10, S * 6 / 10, S * 6 / 10); }, function () { ellipseSel(S * 3 / 10, S * 3 / 10, S * 9 / 10, S * 9 / 10, true); });
  t("Select > Modify > Expand 50 px", selectionUnion, function () { doc.selection.expand(UnitValue(50, "px")); });
  t("Select > Modify > Contract 50 px", selectionUnion, function () { doc.selection.contract(UnitValue(50, "px")); });
  t("Select > Modify > Feather 30 px", selectionUnion, function () { doc.selection.feather(UnitValue(30, "px")); });
  t("Select > Modify > Border 40 px", selectionUnion, function () { doc.selection.selectBorder(UnitValue(40, "px")); });
  t("Select > Modify > Smooth 10 px", selectionUnion, function () { doc.selection.smooth(10); });
  t("Quick Mask enter", selectionUnion, function () { doc.quickMaskMode = true; });
  t("Quick Mask exit", function () { selectionUnion(); doc.quickMaskMode = true; }, function () { doc.quickMaskMode = false; });
  t("magic wand contiguous (sky)", null, function () { wand(10, 10, 32, true, false); });
  t("Select Subject (Photoshop: ML model)", null, function () {
    var d = new ActionDescriptor();
    d.putBoolean(sTID("sampleAllLayers"), false);
    executeAction(sTID("autoCutout"), d, DialogModes.NO);
  });
}
if (PHASES.indexOf("2") >= 0) {
  function ellipseRect(cx, cy, rx, ry) { ellipseSel(cx - rx, cy - ry, cx + rx, cy + ry, false); }
  function caFill() {
    var d = new ActionDescriptor();
    d.putEnumerated(cTID("Usng"), cTID("FlCn"), sTID("contentAware"));
    d.putUnitDouble(cTID("Opct"), cTID("#Prc"), 100);
    d.putEnumerated(cTID("Md  "), cTID("BlnM"), cTID("Nrml"));
    executeAction(cTID("Fl  "), d, DialogModes.NO);
  }
  // Same spots as pixelforge --bench-tools: a 300 px blemish and a 1 MP region.
  t("content-aware fill, 300 px circle (spot-heal equivalent)", function () { ellipseRect(S * 0.5, S * 0.85, 150, 150); }, caFill);
  t("content-aware fill, 1 MP ellipse (content-aware patch equivalent)", function () { ellipseRect(S * 0.45 + 500, S * 0.75 + 500, 500, 500); }, caFill);
  // Probe for scripting hooks of the region tools.
  var probes = [["Patch tool (patchSelection)", "patchSelection"], ["Content-Aware Move (contentAwareMove)", "contentAwareMove"], ["Red Eye (redEyeTool)", "redEyeTool"]];
  for (var i = 0; i < probes.length; i++) {
    t(probes[i][0], function () { ellipseRect(S * 0.5, S * 0.85, 150, 150); }, (function (id) { return function () { executeAction(sTID(id), new ActionDescriptor(), DialogModes.NO); }; })(probes[i][1]));
  }
}
if (PHASES.indexOf("3") >= 0) {
  function grad(type) {
    var d = new ActionDescriptor();
    var f = new ActionDescriptor(); f.putUnitDouble(cTID("Hrzn"), cTID("#Pxl"), S * 0.3); f.putUnitDouble(cTID("Vrtc"), cTID("#Pxl"), S * 0.3);
    d.putObject(cTID("From"), cTID("Pnt "), f);
    var to = new ActionDescriptor(); to.putUnitDouble(cTID("Hrzn"), cTID("#Pxl"), S * 0.7); to.putUnitDouble(cTID("Vrtc"), cTID("#Pxl"), S * 0.6);
    d.putObject(cTID("T   "), cTID("Pnt "), to);
    d.putEnumerated(cTID("Type"), cTID("GrdT"), cTID(type));
    d.putBoolean(cTID("Dthr"), true);
    d.putBoolean(cTID("UsMs"), true);
    var g = new ActionDescriptor();
    g.putString(cTID("Nm  "), "Spectrum");
    g.putEnumerated(cTID("GrdF"), cTID("GrdF"), cTID("CstS"));
    g.putDouble(cTID("Intr"), 4096);
    var clrs = new ActionList();
    var spec = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0]];
    for (var k = 0; k < spec.length; k++) {
      var c = new ActionDescriptor(); var rgb = new ActionDescriptor();
      rgb.putDouble(cTID("Rd  "), spec[k][0]); rgb.putDouble(cTID("Grn "), spec[k][1]); rgb.putDouble(cTID("Bl  "), spec[k][2]);
      c.putObject(cTID("Clr "), cTID("RGBC"), rgb);
      c.putEnumerated(cTID("Type"), cTID("Clry"), cTID("UsrS"));
      c.putInteger(cTID("Lctn"), Math.round(4096 * k / 6)); c.putInteger(cTID("Mdpn"), 50);
      clrs.putObject(cTID("Clrt"), c);
    }
    g.putList(cTID("Clrs"), clrs);
    var trns = new ActionList();
    for (var k2 = 0; k2 < 2; k2++) {
      var tr = new ActionDescriptor(); tr.putUnitDouble(cTID("Opct"), cTID("#Prc"), 100); tr.putInteger(cTID("Lctn"), k2 * 4096); tr.putInteger(cTID("Mdpn"), 50);
      trns.putObject(cTID("TrnS"), tr);
    }
    g.putList(cTID("Trns"), trns);
    d.putObject(cTID("Grad"), cTID("Grdn"), g);
    executeAction(cTID("Grdn"), d, DialogModes.NO);
  }
  var types = [["Linear", "Lnr "], ["Radial", "Rdl "], ["Angle", "Angl"], ["Reflected", "Rflc"], ["Diamond", "Dmnd"]];
  for (var i = 0; i < types.length; i++) {
    t("gradient " + types[i][0] + " (full image)", null, (function (ty) { return function () { grad(ty); }; })(types[i][1]));
  }
  t("paint bucket (sky, contiguous, tol 32)", null, function () {
    var d = new ActionDescriptor();
    var p = new ActionDescriptor(); p.putUnitDouble(cTID("Hrzn"), cTID("#Pxl"), 10); p.putUnitDouble(cTID("Vrtc"), cTID("#Pxl"), 10);
    d.putObject(cTID("From"), cTID("Pnt "), p);
    d.putInteger(cTID("Tlrn"), 32);
    d.putEnumerated(cTID("Usng"), cTID("FlCn"), cTID("FrgC"));
    d.putBoolean(cTID("AntA"), true);
    d.putBoolean(cTID("Cntg"), true);
    executeAction(cTID("Fl  "), d, DialogModes.NO);
  });
  t("magic eraser equivalent (wand sky + delete)", null, function () {
    wand(10, 10, 32, true, false);
    executeAction(cTID("Dlt "), new ActionDescriptor(), DialogModes.NO);
  });
}
if (PHASES.indexOf("4") >= 0) {
  var PARA = "";
  for (var k = 0; k < 24; k++) PARA += "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow! ";
  function paraLayer() {
    var l = doc.artLayers.add();
    l.kind = LayerKind.TEXT;
    var ti = l.textItem;
    ti.kind = TextType.PARAGRAPHTEXT;
    ti.position = [UnitValue(S * 0.1, "px"), UnitValue(S * 0.1, "px")];
    ti.width = UnitValue(2400, "px");
    ti.height = UnitValue(3000, "px");
    ti.size = UnitValue(36, "px");
    ti.contents = PARA;
    return l;
  }
  t("text: create 2000-glyph paragraph layer", null, function () { paraLayer(); });
  var tl;
  t("text: one-character edit of the paragraph (contents setter)", function () { tl = paraLayer(); }, function () { tl.textItem.contents = PARA.substring(0, 1000) + "x" + PARA.substring(1000); });
  function ellipseShape(x0, y0, x1, y1) {
    var d = new ActionDescriptor(); var r = new ActionReference(); r.putClass(sTID("contentLayer")); d.putReference(cTID("null"), r);
    var ld = new ActionDescriptor(); var col = new ActionDescriptor(); var c = new ActionDescriptor();
    c.putDouble(cTID("Rd  "), 200); c.putDouble(cTID("Grn "), 60); c.putDouble(cTID("Bl  "), 40);
    col.putObject(cTID("Clr "), cTID("RGBC"), c);
    ld.putObject(cTID("Type"), sTID("solidColorLayer"), col);
    var e = new ActionDescriptor();
    e.putUnitDouble(cTID("Top "), cTID("#Pxl"), y0); e.putUnitDouble(cTID("Left"), cTID("#Pxl"), x0);
    e.putUnitDouble(cTID("Btom"), cTID("#Pxl"), y1); e.putUnitDouble(cTID("Rght"), cTID("#Pxl"), x1);
    ld.putObject(cTID("Shp "), cTID("Elps"), e);
    d.putObject(cTID("Usng"), sTID("contentLayer"), ld);
    executeAction(cTID("Mk  "), d, DialogModes.NO);
    return doc.activeLayer;
  }
  t("shape: create 9000 px ellipse layer", null, function () { ellipseShape(S * 0.1, S * 0.1, S * 0.1 + 9000, S * 0.1 + 6300); });
  var sl;
  t("shape: resize ellipse layer 110% (one frame)", function () { sl = ellipseShape(S * 0.1, S * 0.1, S * 0.1 + 8000, S * 0.1 + 5600); }, function () { sl.resize(110, 110, AnchorPosition.TOPLEFT); });
  function ellipsePath() {
    var pts = [], cx = S * 0.5, cy = S * 0.5, rx = S * 0.3, ry = S * 0.2, k = 0.5523;
    var a = [[cx + rx, cy, 0, ry * k], [cx, cy + ry, -rx * k, 0], [cx - rx, cy, 0, -ry * k], [cx, cy - ry, rx * k, 0]];
    for (var i = 0; i < 4; i++) {
      var p = new PathPointInfo(); p.kind = PointKind.SMOOTHPOINT;
      p.anchor = [a[i][0], a[i][1]];
      p.leftDirection = [a[i][0] + a[i][2], a[i][1] + a[i][3]];
      p.rightDirection = [a[i][0] - a[i][2], a[i][1] - a[i][3]];
      pts.push(p);
    }
    var sp = new SubPathInfo(); sp.closed = true; sp.operation = ShapeOperation.SHAPEXOR; sp.entireSubPath = pts;
    return doc.pathItems.add("E" + Math.random(), [sp]);
  }
  var pi;
  t("path -> selection (ellipse 9800 x 6500)", function () { pi = ellipsePath(); }, function () { pi.makeSelection(0, true, SelectionType.REPLACE); });
  function starPath() {
    var pts = [];
    for (var i = 0; i < 10; i++) {
      var ang = i * Math.PI / 5 - Math.PI / 2, r = (i % 2 == 0) ? 1500 : 750;
      var p = new PathPointInfo(); p.kind = PointKind.CORNERPOINT;
      p.anchor = [S * 0.5 + r * Math.cos(ang), S * 0.5 + r * Math.sin(ang)];
      p.leftDirection = p.anchor; p.rightDirection = p.anchor;
      pts.push(p);
    }
    var sp = new SubPathInfo(); sp.closed = true; sp.operation = ShapeOperation.SHAPEXOR; sp.entireSubPath = pts;
    return doc.pathItems.add("S" + Math.random(), [sp]);
  }
  t("stroke path (star, brush)", function () { pi = starPath(); app.activeDocument.activeLayer = doc.artLayers[doc.artLayers.length - 1]; }, function () { pi.strokePath(ToolType.BRUSH); });
}
if (PHASES.indexOf("5") >= 0) {
  t("color sampler: add + read RGB", null, function () {
    var cs = doc.colorSamplers.add([UnitValue(S * 0.3, "px"), UnitValue(S * 0.5, "px")]);
    var r = cs.color.rgb.red; cs.remove();
  });
  t("crop (tile-aligned), 268 MP -> 205 MP", null, function () { doc.crop([UnitValue(1024, "px"), UnitValue(512, "px"), UnitValue(S - 1024, "px"), UnitValue(S - 1536, "px")]); });
  t("crop (unaligned), 268 MP -> 205 MP", null, function () { doc.crop([UnitValue(1000, "px"), UnitValue(517, "px"), UnitValue(S - 1048, "px"), UnitValue(S - 1531, "px")]); });
  t("straighten 3.5 deg (rotate canvas + crop)", null, function () {
    var a = 3.5 * Math.PI / 180, k = 1 / (Math.cos(a) + Math.sin(a));
    doc.rotateCanvas(3.5);
    var W = doc.width.as("px"), H = doc.height.as("px"), w = S * k;
    doc.crop([UnitValue((W - w) / 2, "px"), UnitValue((H - w) / 2, "px"), UnitValue((W + w) / 2, "px"), UnitValue((H + w) / 2, "px")]);
  });
  t("free transform: scale layer 90%", function () { doc.activeLayer.isBackgroundLayer = false; }, function () { doc.activeLayer.resize(90, 90, AnchorPosition.MIDDLECENTER); });
}
doc.close(SaveOptions.DONOTSAVECHANGES);
out.join("\n");

// Photoshop side of the gap-closing comparisons (tools/photoshop_gaps.sh).
// Globals set by the shell script: PHOTOS (copy of testdata/photos in /tmp,
// which Photoshop may read without folder permissions), OUT (output dir),
// PHASES (letters, e.g. "A,B"). Timings are best of two with $.hiresTimer.
// Outputs are PNGs the Rust side compares (IoU / SSIM) plus a timing log.
#target photoshop
app.displayDialogs = DialogModes.NO;
app.preferences.rulerUnits = Units.PIXELS;
var out = [];
function cTID(s) { return charIDToTypeID(s); }
function sTID(s) { return stringIDToTypeID(s); }
function savePNG(doc, path) {
  var o = new PNGSaveOptions(); o.compression = 1;
  doc.saveAs(new File(path), o, true, Extension.LOWERCASE);
}
function color(r, g, b) { var c = new SolidColor(); c.rgb.red = r; c.rgb.green = g; c.rgb.blue = b; return c; }
function ellipseSel(cx, cy, rx, ry) {
  var d = new ActionDescriptor();
  var r = new ActionReference(); r.putProperty(cTID("Chnl"), cTID("fsel"));
  d.putReference(cTID("null"), r);
  var e = new ActionDescriptor();
  e.putUnitDouble(cTID("Top "), cTID("#Pxl"), cy - ry);
  e.putUnitDouble(cTID("Left"), cTID("#Pxl"), cx - rx);
  e.putUnitDouble(cTID("Btom"), cTID("#Pxl"), cy + ry);
  e.putUnitDouble(cTID("Rght"), cTID("#Pxl"), cx + rx);
  d.putObject(cTID("T   "), cTID("Elps"), e);
  d.putBoolean(cTID("AntA"), true);
  executeAction(cTID("setd"), d, DialogModes.NO);
}
function caFill() {
  var d = new ActionDescriptor();
  d.putEnumerated(cTID("Usng"), cTID("FlCn"), sTID("contentAware"));
  d.putUnitDouble(cTID("Opct"), cTID("#Prc"), 100);
  d.putEnumerated(cTID("Md  "), cTID("BlnM"), cTID("Nrml"));
  executeAction(cTID("Fl  "), d, DialogModes.NO);
}
// Times fn twice from the same history state; keeps the second result.
function timed(doc, name, prep, fn) {
  var base = doc.historyStates.length - 1, best = 1e15;
  for (var k = 0; k < 2; k++) {
    doc.activeHistoryState = doc.historyStates[base];
    if (prep) prep();
    $.hiresTimer; fn(); var us = $.hiresTimer;
    if (us < best) best = us;
  }
  out.push(name + "\t" + (best / 1000).toFixed(1));
  return base;
}
// The selection as a black/white PNG.
function saveSelection(doc, path) {
  var ch = doc.channels.add(); doc.selection.store(ch);
  var lay = doc.artLayers.add();
  doc.selection.selectAll(); doc.selection.fill(color(0, 0, 0));
  doc.selection.load(ch); doc.selection.fill(color(255, 255, 255));
  savePNG(doc, path);
  lay.remove(); ch.remove();
}

// Leftovers of an interrupted run (e.g. the photo still open from a failed
// export) would trigger "disk copy changed" dialogs: close them first.
for (var di = app.documents.length - 1; di >= 0; di--) {
  var dn = app.documents[di].name;
  if (dn.indexOf(".jpg") >= 0 || dn == "warp" || dn == "paths" || dn == "text" || dn == "tmp") app.documents[di].close(SaveOptions.DONOTSAVECHANGES);
}
var PH = "," + PHASES + ",";
if (PH.indexOf(",A,") >= 0) {
  var photos = ["portrait_hair", "dog_grass", "butterfly"];
  // Content-aware fill holes (cx, cy, rx, ry), shared with `--bench-tools quality`.
  var holes = {
    dog_grass: [["300px", 400, 2100, 150, 150], ["1MP", 3300, 1900, 531, 600], ["daisies", 1720, 430, 240, 90]],
    butterfly: [["300px", 2200, 420, 150, 150]],
    portrait_hair: [["300px", 500, 600, 150, 150]]
  };
  for (var i = 0; i < photos.length; i++) {
    var photo = photos[i];
    var doc = app.open(new File(PHOTOS + "/" + photo + ".jpg"));
    var base = timed(doc, "Select Subject " + photo, null, function () {
      var d = new ActionDescriptor(); d.putBoolean(sTID("sampleAllLayers"), false);
      executeAction(sTID("autoCutout"), d, DialogModes.NO);
    });
    saveSelection(doc, OUT + "/" + photo + "_subject.png");
    var hs = holes[photo] || [];
    for (var j = 0; j < hs.length; j++) {
      var h = hs[j];
      doc.activeHistoryState = doc.historyStates[base];
      timed(doc, "Content-Aware Fill " + h[0] + " " + photo, function () { ellipseSel(h[1], h[2], h[3], h[4]); }, caFill);
      doc.selection.deselect();
      savePNG(doc, OUT + "/" + photo + "_caf_" + h[0] + ".png");
    }
    doc.close(SaveOptions.DONOTSAVECHANGES);
  }
}
if (PH.indexOf(",B,") >= 0) {
  // Warp reference shapes: three lines of full-block glyphs (a grid of
  // corners spanning the text box), each warp style at several settings,
  // converted to a work path whose points are dumped for the Rust side.
  var wd = app.documents.add(UnitValue(3000, "px"), UnitValue(1600, "px"), 72, "warp", NewDocumentMode.RGB, DocumentFill.WHITE);
  var styles = ["NONE", "ARC", "ARCLOWER", "ARCUPPER", "ARCH", "BULGE", "SHELLLOWER", "SHELLUPPER", "FLAG", "WAVE", "FISH", "RISE", "FISHEYE", "INFLATE", "SQUEEZE", "TWIST"];
  var settings = [[50, 0, 0], [-50, 0, 0], [50, 40, 0], [50, 0, 40], [100, 0, 0], [100, 40, 0], [100, 0, 40], [-50, 40, 0], [-50, 0, 40], [50, -40, 0], [50, 0, -40], [50, 80, 0], [50, 0, 80], [50, 40, 40], [25, 0, 0], [75, 0, 0]];
  var lines = [];
  for (var si = 0; si < styles.length; si++) {
    for (var k = 0; k < settings.length; k++) {
      if (si == 0 && k > 0) continue;
      var lay = wd.artLayers.add(); lay.kind = LayerKind.TEXT;
      var ti = lay.textItem;
      ti.font = "Menlo-Regular"; ti.size = 120; ti.color = (function () { var c = new SolidColor(); c.rgb.red = 0; c.rgb.green = 0; c.rgb.blue = 0; return c; })();
      ti.contents = "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\r\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\r\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588";
      ti.position = [UnitValue(600, "px"), UnitValue(500, "px")];
      ti.warpStyle = WarpStyle[styles[si]];
      if (si > 0) {
        ti.warpBend = settings[k][0];
        ti.warpHorizontalDistortion = settings[k][1];
        ti.warpVerticalDistortion = settings[k][2];
        ti.warpDirection = Direction.HORIZONTAL;
      }
      ti.createPath();
      var pi = wd.pathItems.getByName("Work Path");
      var rec = [styles[si], settings[k][0], settings[k][1], settings[k][2]].join(" ");
      for (var sp = 0; sp < pi.subPathItems.length; sp++) {
        var pts = pi.subPathItems[sp].pathPoints, a = [];
        for (var q = 0; q < pts.length; q++) {
          var P = pts[q];
          var v = [P.anchor[0], P.anchor[1], P.leftDirection[0], P.leftDirection[1], P.rightDirection[0], P.rightDirection[1]];
          for (var vi = 0; vi < v.length; vi++) v[vi] = Math.round(v[vi] * 100) / 100;
          a.push(v.join(","));
        }
        lines.push(rec + " | " + a.join(";"));
      }
      pi.remove();
      lay.remove();
    }
  }
  var f = new File(OUT + "/warps.txt"); f.open("w"); f.write(lines.join("\n")); f.close();
  out.push("warp shapes\t" + lines.length + " subpaths");
  wd.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",BT,") >= 0) {
  // Styled text timings (ActionManager is the only scripting hook for
  // style runs): the same 2000-glyph paragraph and 51 runs as
  // `--bench-tools type`, in a 2400 × 3000 box.
  var td = app.documents.add(UnitValue(3000, "px"), UnitValue(3000, "px"), 72, "text", NewDocumentMode.RGB, DocumentFill.WHITE);
  var unit = "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow! ";
  var para = ""; for (var i = 0; i < 24; i++) para += unit;
  // runs: [from, to, size, r, g, b, tracking]
  function makeRuns(n, styledColor, shiftAt, shiftBy) {
    var rs = [], at = 0;
    for (var k = 0; k < 25; k++) {
      var a = k * 80 + 20, b = a + 40;
      if (a > at) rs.push([at, a, 36, 0, 0, 0, 0]);
      rs.push([a, b, 44, styledColor[0], styledColor[1], styledColor[2], 50]);
      at = b;
    }
    if (at < n) rs.push([at, n, 36, 0, 0, 0, 0]);
    if (shiftBy) for (var q = 0; q < rs.length; q++) { if (rs[q][0] > shiftAt) rs[q][0] += shiftBy; if (rs[q][1] >= shiftAt) rs[q][1] += shiftBy; }
    return rs;
  }
  function textDesc(txt, runs) {
    var d = new ActionDescriptor();
    d.putString(sTID("textKey"), txt);
    var click = new ActionDescriptor();
    click.putUnitDouble(cTID("Hrzn"), cTID("#Pxl"), 300); click.putUnitDouble(cTID("Vrtc"), cTID("#Pxl"), 300);
    d.putObject(cTID("TxtC"), cTID("Pnt "), click);
    var shape = new ActionList(), sd = new ActionDescriptor();
    sd.putEnumerated(sTID("textType"), sTID("textType"), sTID("box"));
    sd.putEnumerated(cTID("Ornt"), cTID("Ornt"), cTID("Hrzn"));
    var bx = new ActionDescriptor();
    bx.putUnitDouble(cTID("Top "), cTID("#Pnt"), 0); bx.putUnitDouble(cTID("Left"), cTID("#Pnt"), 0);
    bx.putUnitDouble(cTID("Btom"), cTID("#Pnt"), 3000); bx.putUnitDouble(cTID("Rght"), cTID("#Pnt"), 2400);
    sd.putObject(sTID("bounds"), cTID("Rctn"), bx);
    shape.putObject(sTID("textShape"), sd);
    d.putList(sTID("textShape"), shape);
    var list = new ActionList();
    for (var k = 0; k < runs.length; k++) {
      var r = runs[k], rd = new ActionDescriptor(), st = new ActionDescriptor(), c = new ActionDescriptor();
      rd.putInteger(cTID("From"), r[0]); rd.putInteger(cTID("T   "), r[1]);
      st.putString(sTID("fontPostScriptName"), "Helvetica");
      st.putUnitDouble(cTID("Sz  "), cTID("#Pnt"), r[2]);
      st.putInteger(sTID("tracking"), r[6]);
      c.putDouble(cTID("Rd  "), r[3]); c.putDouble(cTID("Grn "), r[4]); c.putDouble(cTID("Bl  "), r[5]);
      st.putObject(cTID("Clr "), cTID("RGBC"), c);
      rd.putObject(cTID("TxtS"), cTID("TxtS"), st);
      list.putObject(cTID("Txtt"), rd);
    }
    d.putList(cTID("Txtt"), list);
    return d;
  }
  function makeText(txt, runs) {
    var d = new ActionDescriptor(), r = new ActionReference();
    r.putClass(cTID("TxLr")); d.putReference(cTID("null"), r);
    d.putObject(cTID("Usng"), cTID("TxLr"), textDesc(txt, runs));
    executeAction(cTID("Mk  "), d, DialogModes.NO);
  }
  function setText(txt, runs) {
    var d = new ActionDescriptor(), r = new ActionReference();
    r.putEnumerated(cTID("TxLr"), cTID("Ordn"), cTID("Trgt")); d.putReference(cTID("null"), r);
    d.putObject(cTID("T   "), cTID("TxLr"), textDesc(txt, runs));
    executeAction(cTID("setd"), d, DialogModes.NO);
  }
  var runs = makeRuns(para.length, [200, 20, 40]);
  timed(td, "Text: create 2000-glyph layer with " + runs.length + " runs", null, function () { makeText(para, runs); });
  var runs2 = makeRuns(para.length, [200, 20, 40]); runs2[21][3] = 20; runs2[21][4] = 60; runs2[21][5] = 200;
  timed(td, "Text: edit one run (colour)", null, function () { setText(para, runs2); });
  var mid = Math.floor(para.length / 2);
  var para2 = para.substring(0, mid) + "x" + para.substring(mid);
  var runs3 = makeRuns(para.length, [200, 20, 40], mid, 1);
  timed(td, "Text: insert one character", null, function () { setText(para2, runs3); });
  td.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",C,") >= 0) {
  // Path operations: two 500-anchor smooth closed paths (the same curves as
  // `--bench-tools vector` / `--compare-gaps C`), the second one combined
  // with each operation; Make Selection (feather 0, antialiased) timed, and
  // the selected area measured from the saved channel's histogram.
  var cd = app.documents.add(UnitValue(5200, "px"), UnitValue(5200, "px"), 72, "paths", NewDocumentMode.RGB, DocumentFill.WHITE);
  function mul32(a, b) {
    var lo = (a & 0xffff) * b;
    var hi = ((a >>> 16) * b) & 0xffff;
    return ((hi << 16) + lo) >>> 0;
  }
  function curvy(cx, cy, r, seed, op) {
    var n = 500, pts = [];
    for (var i = 0; i < n; i++) {
      var t = 2 * Math.PI * i / n;
      var h = (((mul32(i, 2654435761) + seed) >>> 0) >>> 8) / 16777216;
      var rr = r + 60 * (h - 0.5) + 40 * Math.sin(7 * t);
      pts.push([cx + rr * Math.cos(t), cy + rr * Math.sin(t)]);
    }
    var arr = [];
    for (var k = 0; k < n; k++) {
      var nx = pts[(k + 1) % n], pv = pts[(k + n - 1) % n];
      var dx = (nx[0] - pv[0]) / 6, dy = (nx[1] - pv[1]) / 6;
      var pp = new PathPointInfo();
      pp.kind = PointKind.SMOOTHPOINT;
      pp.anchor = pts[k];
      // Photoshop's left / right directions are out / in along the path.
      pp.leftDirection = [pts[k][0] + dx, pts[k][1] + dy];
      pp.rightDirection = [pts[k][0] - dx, pts[k][1] - dy];
      arr.push(pp);
    }
    var sp = new SubPathInfo();
    sp.closed = true;
    sp.operation = op;
    sp.entireSubPath = arr;
    return sp;
  }
  var ops = [["Combine", ShapeOperation.SHAPEADD], ["Subtract", ShapeOperation.SHAPESUBTRACT], ["Intersect", ShapeOperation.SHAPEINTERSECT], ["Exclude", ShapeOperation.SHAPEXOR]];
  var lines = [];
  for (var oi = 0; oi < ops.length; oi++) {
    var a = curvy(2500, 2500, 2000, 1, ShapeOperation.SHAPEADD);
    var b = curvy(2650, 2600, 2000, 2, ops[oi][1]);
    var best = 1e15;
    var pi;
    for (var rep = 0; rep < 2; rep++) {
      pi = cd.pathItems.add("ops" + oi + "_" + rep, [a, b]);
      $.hiresTimer;
      pi.makeSelection(0, true, SelectionType.REPLACE);
      var us = $.hiresTimer;
      if (us < best) best = us;
      if (rep == 0) pi.remove();
    }
    var ch = cd.channels.add();
    cd.selection.store(ch);
    var hist = ch.histogram;
    var area = 0;
    for (var v = 1; v < 256; v++) area += hist[v] * v / 255;
    ch.remove();
    cd.selection.deselect();
    pi.remove();
    lines.push(ops[oi][0] + "\t" + area.toFixed(1) + "\t" + (best / 1000).toFixed(2));
    out.push("Path op " + ops[oi][0] + " + Make Selection\t" + (best / 1000).toFixed(1));
  }
  var f = new File(OUT + "/paths_C.txt"); f.open("w"); f.write(lines.join("\n")); f.close();
  cd.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",D,") >= 0) {
  // Web export of the dog photo at 1920 × 1280 (the resized source is saved
  // losslessly so both apps encode the same pixels). Photoshop 2026's legacy
  // Save for Web cancels under scripting, so its scriptable equivalents are
  // timed: PNG-8 = Image ▸ Mode ▸ Indexed Color (local adaptive, 128 colours,
  // diffusion dither) + Save As PNG; PNG-24 = Save As PNG; JPEG = Save As
  // JPEG quality 6 (of 12). Best of two; files go to OUT for SSIM.
  var src = app.open(new File(PHOTOS + "/dog_grass.jpg"));
  src.resizeImage(UnitValue(1920, "px"), UnitValue(1280, "px"), 72, ResampleMethod.BICUBIC);
  savePNG(src, OUT + "/web_src.png");
  var lines = [];
  function bench(name, ext, fn) {
    var best = 1e15, path = OUT + "/web_" + name + "." + ext;
    for (var rep = 0; rep < 2; rep++) {
      // Duplicate the RGB source explicitly (an Indexed leftover would make
      // the JPEG save stop on a "cannot be saved as jpeg" dialog).
      app.activeDocument = src;
      var d = src.duplicate("tmp");
      if (d.mode != DocumentMode.RGB) d.changeMode(ChangeMode.RGB);
      var f = new File(path);
      if (f.exists) f.remove();
      $.hiresTimer;
      fn(d, f);
      var us = $.hiresTimer;
      if (us < best) best = us;
      d.close(SaveOptions.DONOTSAVECHANGES);
    }
    var size = new File(path).length;
    lines.push(name + "\t" + size + "\t" + (best / 1000).toFixed(1));
    out.push("Web export " + name + "\t" + (best / 1000).toFixed(1));
  }
  bench("png8", "png", function (d, f) {
    var o = new IndexedConversionOptions();
    o.palette = Palette.LOCALADAPTIVE; o.colors = 128; o.dither = Dither.DIFFUSION; o.ditherAmount = 100; o.forced = ForcedColors.NONE; o.transparency = false;
    d.changeMode(ChangeMode.INDEXEDCOLOR, o);
    var p = new PNGSaveOptions(); p.compression = 9;
    d.saveAs(f, p, true, Extension.LOWERCASE);
  });
  bench("png24", "png", function (d, f) {
    var p = new PNGSaveOptions(); p.compression = 9;
    d.saveAs(f, p, true, Extension.LOWERCASE);
  });
  bench("jpeg60", "jpg", function (d, f) {
    var j = new JPEGSaveOptions(); j.quality = 6; j.formatOptions = FormatOptions.OPTIMIZEDBASELINE; j.embedColorProfile = false;
    d.saveAs(f, j, true, Extension.LOWERCASE);
  });
  var lf = new File(OUT + "/web_D.txt"); lf.open("w"); lf.write(lines.join("\n")); lf.close();
  src.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",E,") >= 0) {
  // Rotated crop: Image ▸ Rotate Canvas 8° (Bicubic Automatic) and the
  // centred 3200 × 2000 crop — what PixelForge's rotated crop produces.
  var doc = app.open(new File(PHOTOS + "/dog_grass.jpg"));
  var base = timed(doc, "Rotate canvas 8 + crop 3200x2000", null, function () {
    doc.rotateCanvas(8);
    var cx = doc.width.as("px") / 2, cy = doc.height.as("px") / 2;
    doc.crop([cx - 1600, cy - 1000, cx + 1600, cy + 1000]);
  });
  savePNG(doc, OUT + "/crop_rot8.png");
  doc.activeHistoryState = doc.historyStates[base];
  // Crop with Delete Cropped Pixels off (the Crop event's "delete" flag).
  try {
    timed(doc, "Crop, Delete Cropped Pixels off", null, function () {
      var d = new ActionDescriptor(), r = new ActionDescriptor();
      r.putUnitDouble(cTID("Top "), cTID("#Pxl"), 300); r.putUnitDouble(cTID("Left"), cTID("#Pxl"), 400);
      r.putUnitDouble(cTID("Btom"), cTID("#Pxl"), 2300); r.putUnitDouble(cTID("Rght"), cTID("#Pxl"), 3600);
      d.putObject(cTID("T   "), cTID("Rctn"), r);
      d.putUnitDouble(cTID("Angl"), cTID("#Ang"), 0);
      d.putBoolean(sTID("delete"), false);
      executeAction(cTID("Crop"), d, DialogModes.NO);
    });
  } catch (e) { out.push("Crop, Delete Cropped Pixels off\tfailed: " + e); }
  doc.activeHistoryState = doc.historyStates[base];
  // Content-Aware: the photo turned 8° on a layer; its empty corners
  // (grown 2 px) filled with Content-Aware Fill.
  doc.activeLayer.isBackgroundLayer = false;
  doc.activeLayer.rotate(8, AnchorPosition.MIDDLECENTER);
  timed(doc, "Content-aware fill: corners of an 8 turn", function () {
    var d = new ActionDescriptor(), r = new ActionReference(), t = new ActionReference();
    r.putProperty(cTID("Chnl"), cTID("fsel")); d.putReference(cTID("null"), r);
    t.putEnumerated(cTID("Chnl"), cTID("Chnl"), cTID("Trsp")); d.putReference(cTID("T   "), t);
    executeAction(cTID("setd"), d, DialogModes.NO);
    doc.selection.invert(); doc.selection.expand(UnitValue(2, "px"));
  }, function () { caFill(); });
  savePNG(doc, OUT + "/caf_rot8.png");
  doc.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",F,") >= 0) {
  // Info panel colour readouts: Photoshop's Lab (D50), HSB and CMYK (its
  // working CMYK, U.S. Web Coated (SWOP) v2 by default) for a 17³ sRGB grid
  // plus 200 random colours. One line: r g b C M Y K L a b H S B.
  var lines = [];
  var skipProbes = (typeof PF_SKIP_PROBES != "undefined");
  var d0 = app.documents.add(UnitValue(64, "px"), UnitValue(64, "px"), 72, "tmp", NewDocumentMode.RGB);
  var t0 = $.hiresTimer;
  function probe(r, g, b) {
    var c = new SolidColor(); c.rgb.red = r; c.rgb.green = g; c.rgb.blue = b;
    lines.push([r, g, b, c.cmyk.cyan, c.cmyk.magenta, c.cmyk.yellow, c.cmyk.black, c.lab.l, c.lab.a, c.lab.b, c.hsb.hue, c.hsb.saturation, c.hsb.brightness].join(" "));
  }
  if (!skipProbes) for (var ri = 0; ri < 17; ri++) for (var gi = 0; gi < 17; gi++) for (var bi = 0; bi < 17; bi++) {
    probe(Math.min(255, ri * 16), Math.min(255, gi * 16), Math.min(255, bi * 16));
  }
  var seed = 12345;
  function rnd() { seed = (seed * 1103515245 + 12345) % 2147483648; return Math.floor(seed / 2147483648 * 256); }
  if (!skipProbes) {
    for (var k = 0; k < 200; k++) probe(rnd(), rnd(), rnd());
    out.push("Info colour probes (" + lines.length + ")\t" + (($.hiresTimer) / 1000).toFixed(1));
    var lf = new File(OUT + "/info_F.txt"); lf.open("w"); lf.write(lines.join("\n")); lf.close();
  }
  d0.close(SaveOptions.DONOTSAVECHANGES);
  // Measurement Log: an elliptical selection (100,100)–(500,500) on a
  // 700 × 600 document filled with gray 100.
  var dm = app.documents.add(UnitValue(700, "px"), UnitValue(600, "px"), 72, "tmp", NewDocumentMode.RGB);
  dm.selection.selectAll(); dm.selection.fill(color(100, 100, 100)); dm.selection.deselect();
  ellipseSel(300, 300, 200, 200);
  try { app.measurementLog.deleteMeasurements(MeasurementRange.ALLMEASUREMENTS); } catch (e) {}
  $.hiresTimer;
  // (Photoshop rejects explicit data-point lists here: record its defaults.)
  dm.recordMeasurements(MeasurementSource.MEASURESELECTION);
  out.push("Record Measurements (selection)\t" + ($.hiresTimer / 1000).toFixed(1));
  // Gray Value (Mean) of flat fills: Photoshop's gray scale (its Gray
  // working space, Dot Gain 20 %) for 17 grays and 4 colours.
  var fills = [];
  for (var v = 0; v <= 256; v += 16) fills.push([Math.min(v, 255), Math.min(v, 255), Math.min(v, 255)]);
  fills.push([255, 0, 0], [0, 255, 0], [0, 0, 255], [200, 150, 50]);
  for (var fi = 0; fi < fills.length; fi++) {
    dm.selection.selectAll(); dm.selection.fill(color(fills[fi][0], fills[fi][1], fills[fi][2]));
    ellipseSel(300, 300, 200, 200);
    dm.recordMeasurements(MeasurementSource.MEASURESELECTION);
  }
  var mf = new File(OUT + "/measure_F.csv");
  if (mf.exists) mf.remove();
  app.measurementLog.exportMeasurements(mf, MeasurementRange.ALLMEASUREMENTS);
  dm.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",G,") >= 0) {
  // Filter Gallery fidelity: every effect at its default settings (an empty
  // descriptor) on the dog photo at 1024 × 683, foreground black and
  // background white (Photoshop's defaults). Saves gallery_src.png and
  // gallery_<n>.png, n = the effect's index in PixelForge's G::ALL order.
  var ids = [["Colored Pencil","ClrP"],["Cutout","Ct  "],["Dry Brush","DryB"],["Film Grain","FlmG"],["Fresco","Frsc"],["Neon Glow","NGlw"],["Paint Daubs","PntD"],["Palette Knife","PltK"],["Plastic Wrap","PlsW"],["Poster Edges","PstE"],["Rough Pastels","RghP"],["Smudge Stick","SmdS"],["Sponge","Spng"],["Underpainting","Undr"],["Watercolor","Wtrc"],["Accented Edges","AccE"],["Angled Strokes","AngS"],["Crosshatch","Crsh"],["Dark Strokes","DrkS"],["Ink Outlines","InkO"],["Spatter","Spt "],["Sprayed Strokes","SprS"],["Sumi-e","Smie"],["Diffuse Glow","DfsG"],["Glass","Gls "],["Ocean Ripple","OcnR"],["Bas Relief","BsRl"],["Chalk & Charcoal","ChlC"],["Charcoal","Chrc"],["Chrome","Chrm"],["Conté Crayon","CntC"],["Graphic Pen","GraP"],["Halftone Pattern","HlfS"],["Note Paper","NtPr"],["Photocopy","Phtc"],["Plaster","Plst"],["Reticulation","Rtcl"],["Stamp","Stmp"],["Torn Edges","TrnE"],["Water Paper","WtrP"],["Glowing Edges","GlwE"],["Craquelure","Crql"],["Grain","Grn "],["Mosaic Tiles","MscT"],["Patchwork","Ptch"],["Stained Glass","StnG"],["Texturizer","Txtz"]];
  var black = new SolidColor(); black.rgb.red = 0; black.rgb.green = 0; black.rgb.blue = 0;
  var white = new SolidColor(); white.rgb.red = 255; white.rgb.green = 255; white.rgb.blue = 255;
  app.foregroundColor = black; app.backgroundColor = white;
  var g = app.open(new File(PHOTOS + "/dog_grass.jpg"));
  g.resizeImage(UnitValue(1024, "px"), UnitValue(683, "px"), 72, ResampleMethod.BICUBIC);
  savePNG(g, OUT + "/gallery_src.png");
  var gbase = g.historyStates.length - 1;
  var only = (typeof PF_ONLY != "undefined") ? PF_ONLY : -1;
  for (var gi = 0; gi < ids.length; gi++) {
    if (only >= 0 && gi != only) continue;
    try {
      g.activeHistoryState = g.historyStates[gbase];
      $.hiresTimer;
      executeAction(charIDToTypeID(ids[gi][1]), new ActionDescriptor(), DialogModes.NO);
      var gus = $.hiresTimer;
      savePNG(g, OUT + "/gallery_" + gi + ".png");
      out.push("Gallery " + ids[gi][0] + "\t" + (gus / 1000).toFixed(1));
    } catch (e) { out.push("Gallery " + ids[gi][0] + "\tERR " + String(e.message).split("\n")[0]); }
  }
  g.close(SaveOptions.DONOTSAVECHANGES);
}
if (PH.indexOf(",GPU,") >= 0) {
  // GPU comparisons (README "GPU acceleration"): the exported scene at
  // 4096² (16.7 MP, outputs saved as gpu_<key>.png) and 16384² (268 MP,
  // timed only). Image Size ▸ Preserve Details enlarges the 8192² scene 2×.
  // Scenes come from `pixelforge --export-scene <side> <PHOTOS>/scene_<side>.tif`.
  // Blur Gallery and Liquify have no scripting hook (their events need the
  // full widget descriptor), so they aren't timed here.
  function num(d, k, v) { d.putDouble(cTID(k), v); }
  function px(d, k, v) { d.putUnitDouble(cTID(k), cTID("#Pxl"), v); }
  function pct(d, k, v) { d.putUnitDouble(cTID(k), cTID("#Prc"), v); }
  var ops = [
    ["gaussian", "Gaussian Blur r=20", function () { var d = new ActionDescriptor(); px(d, "Rds ", 20); executeAction(cTID("GsnB"), d, DialogModes.NO); }],
    ["surface", "Surface Blur r=5 t=15", function () { var d = new ActionDescriptor(); px(d, "Rds ", 5); d.putInteger(cTID("Thsh"), 15); executeAction(sTID("surfaceBlur"), d, DialogModes.NO); }],
    ["median", "Median r=3", function () { var d = new ActionDescriptor(); px(d, "Rds ", 3); executeAction(cTID("Mdn "), d, DialogModes.NO); }],
    ["unsharp", "Unsharp Mask 100 % r=1.5", function () { var d = new ActionDescriptor(); pct(d, "Amnt", 100); px(d, "Rds ", 1.5); d.putInteger(cTID("Thsh"), 0); executeAction(cTID("UnsM"), d, DialogModes.NO); }],
    ["smartsharpen", "Smart Sharpen 100 % r=1.5", function () { var d = new ActionDescriptor(); pct(d, "Amnt", 100); px(d, "Rds ", 1.5); d.putInteger(sTID("noiseReduction"), 10); executeAction(sTID("smartSharpen"), d, DialogModes.NO); }],
    ["lensblur", "Lens Blur (defaults)", function () { executeAction(cTID("Bokh"), new ActionDescriptor(), DialogModes.NO); }],
    ["oilpaint", "Oil Paint (defaults)", function () { executeAction(sTID("oilPaint"), new ActionDescriptor(), DialogModes.NO); }],
    ["cameraraw", "Camera Raw Filter (heavy)", function () {
      var d = new ActionDescriptor();
      // Filter mode with custom settings, as a recorded action has them:
      // without these keys Camera Raw opens its dialog.
      d.putString(cTID("CMod"), "Filter");
      d.putEnumerated(cTID("Sett"), cTID("Sett"), cTID("Cst "));
      d.putEnumerated(cTID("WBal"), cTID("WBal"), cTID("AsSh"));
      num(d, "Ex12", 0.35); d.putInteger(cTID("Cr12"), 18); d.putInteger(cTID("Hi12"), -40); d.putInteger(cTID("Sh12"), 35);
      d.putInteger(cTID("Wh12"), 10); d.putInteger(cTID("Bk12"), -12); d.putInteger(cTID("Cl12"), 30); d.putInteger(cTID("Dhze"), 20);
      d.putInteger(cTID("Vibr"), 20); d.putInteger(cTID("Strt"), 5); d.putInteger(cTID("Shrp"), 40); d.putInteger(cTID("LNR "), 30);
      executeAction(sTID("Adobe Camera Raw Filter"), d, DialogModes.NO);
    }],
    ["cutout", "Gallery Cutout", function () { executeAction(cTID("Ct  "), new ActionDescriptor(), DialogModes.NO); }],
    ["posteredges", "Gallery Poster Edges", function () { executeAction(cTID("PstE"), new ActionDescriptor(), DialogModes.NO); }],
    ["neonglow", "Gallery Neon Glow", function () { executeAction(cTID("NGlw"), new ActionDescriptor(), DialogModes.NO); }]
  ];
  var only = (typeof PF_ONLY != "undefined") ? PF_ONLY : "";
  // Each timing is also appended to timings_GPU_live.txt as it's measured,
  // so a stalled run (a dialog, a crash) keeps what it got.
  function live(line) {
    out.push(line);
    var f = new File(OUT + "/timings_GPU_live.txt");
    f.lineFeed = "Unix"; f.open("a"); f.writeln(line); f.close();
  }
  app.foregroundColor = color(0, 0, 0); app.backgroundColor = color(255, 255, 255);
  var sides = (typeof PF_SIDES != "undefined") ? PF_SIDES : [4096, 16384];
  for (var si = 0; si < sides.length; si++) {
    var side = sides[si];
    var sd = app.open(new File(PHOTOS + "/scene_" + side + ".tif"));
    var sbase = sd.historyStates.length - 1;
    for (var oi = 0; oi < ops.length; oi++) {
      if (only != "" && ops[oi][0] != only) continue;
      var best = 1e30;
      var reps = side > 8192 ? 1 : 2;
      try {
        for (var r = 0; r < reps; r++) {
          sd.activeHistoryState = sd.historyStates[sbase];
          $.hiresTimer;
          ops[oi][2]();
          best = Math.min(best, $.hiresTimer / 1000);
        }
        if (side == 4096) savePNG(sd, OUT + "/gpu_" + ops[oi][0] + ".png");
        live("GPU " + ops[oi][0] + " " + side + "\t" + best.toFixed(1));
      } catch (e) { live("GPU " + ops[oi][0] + " " + side + "\tERR " + String(e.message).split("\n")[0]); }
    }
    sd.close(SaveOptions.DONOTSAVECHANGES);
  }
  if (only == "" || only == "preservedetails") {
    var pd = app.open(new File(PHOTOS + "/scene_8192.tif"));
    $.hiresTimer;
    pd.resizeImage(UnitValue(16384, "px"), UnitValue(16384, "px"), 72, ResampleMethod.PRESERVEDETAILS);
    live("GPU preservedetails 16384\t" + ($.hiresTimer / 1000).toFixed(1));
    pd.close(SaveOptions.DONOTSAVECHANGES);
  }
  if (only == "" || only == "preservedetails" || only == "preservedetails4096") {
    // A PNG: opening a second TIFF after the Camera Raw filter ran made Camera
    // Raw intercept the open with its dialog.
    var pd2 = app.open(new File(PHOTOS + "/scene_2048.png"));
    $.hiresTimer;
    pd2.resizeImage(UnitValue(4096, "px"), UnitValue(4096, "px"), 72, ResampleMethod.PRESERVEDETAILS);
    live("GPU preservedetails 4096\t" + ($.hiresTimer / 1000).toFixed(1));
    savePNG(pd2, OUT + "/gpu_preservedetails.png");
    pd2.close(SaveOptions.DONOTSAVECHANGES);
  }
}
out.join("\n");

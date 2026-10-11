#target photoshop
app.displayDialogs = DialogModes.NO;
var SIDE = 8192;
var out = [];
var doc = app.documents.add(UnitValue(SIDE, "px"), UnitValue(SIDE, "px"), 72, "pf-bench", NewDocumentMode.RGB, DocumentFill.WHITE);
doc.activeLayer.applyClouds();
doc.activeLayer.applyAddNoise(8, NoiseDistribution.UNIFORM, false);
var base = doc.historyStates.length - 1;
function am(id, desc) { executeAction(stringIDToTypeID(id), desc || new ActionDescriptor(), DialogModes.NO); }
function t(name, fn) {
  try {
    var best = 1e12;
    for (var k = 0; k < 2; k++) {
      doc.activeHistoryState = doc.historyStates[base];
      $.hiresTimer;
      fn(doc.activeLayer);
      var us = $.hiresTimer;
      if (us < best) best = us;
    }
    out.push(name + "\t" + (best / 1000).toFixed(1));
  } catch (e) { out.push(name + "\tERR " + e.message); }
}
function d1(k, v) { var d = new ActionDescriptor(); d.putUnitDouble(charIDToTypeID(k), charIDToTypeID("#Pxl"), v); return d; }
t("Brightness/Contrast", function (l) { l.adjustBrightnessContrast(20, 20); });
t("Levels", function (l) { l.adjustLevels(20, 230, 1.2, 0, 255); });
t("Invert", function (l) { l.invert(); });
t("Desaturate", function (l) { l.desaturate(); });
t("Posterize", function (l) { l.posterize(4); });
t("Threshold", function (l) { l.threshold(128); });
t("Equalize", function (l) { l.equalize(); });
t("Auto Contrast", function (l) { l.autoContrast(); });
t("Shadows/Highlights", function (l) { l.shadowHighlight(35, 30, 0, 0, 0, 0, 20, 0, 0); });
t("Average", function (l) { l.applyAverage(); });
t("Blur", function (l) { l.applyBlur(); });
t("Blur More", function (l) { l.applyBlurMore(); });
t("Box Blur", function () { am("boxblur", d1("Rds ", 10)); });
t("Gaussian Blur", function (l) { l.applyGaussianBlur(5); });
t("Gaussian Blur r=120", function (l) { l.applyGaussianBlur(120); });
t("Motion Blur", function (l) { l.applyMotionBlur(0, 40); });
t("Radial Blur", function (l) { l.applyRadialBlur(10, RadialBlurMethod.SPIN, RadialBlurQuality.BEST); });
t("Smart Blur", function (l) { l.applySmartBlur(3, 25, SmartBlurQuality.HIGH, SmartBlurMode.NORMAL); });
t("Surface Blur", function () { var d = d1("Rds ", 5); d.putInteger(charIDToTypeID("Thsh"), 15); am("surfaceBlur", d); });
t("Lens Blur", function (l) { l.applyLensBlur(DepthMapSource.NONE, 0, 0, false, Geometry.HEXAGON, 15, 0, 0, 0, 255, 0, NoiseDistribution.UNIFORM, false); });
t("Pinch", function (l) { l.applyPinch(50); });
t("Polar Coordinates", function (l) { l.applyPolarCoordinates(PolarConversionType.RECTANGULARTOPOLAR); });
t("Ripple", function (l) { l.applyRipple(100, RippleSize.MEDIUM); });
t("Spherize", function (l) { l.applySpherize(100, SpherizeMode.NORMAL); });
t("Twirl", function (l) { l.applyTwirl(50); });
t("ZigZag", function (l) { l.applyZigZag(30, 5, ZigZagType.PONDRIPPLES); });
t("Add Noise", function (l) { l.applyAddNoise(12.5, NoiseDistribution.UNIFORM, false); });
t("Despeckle", function (l) { l.applyDespeckle(); });
t("Dust & Scratches", function (l) { l.applyDustAndScratches(1, 0); });
t("Median", function (l) { l.applyMedianNoise(2); });
t("Mosaic", function () { am("mosaicPlugin", (function(){ var d = new ActionDescriptor(); d.putUnitDouble(charIDToTypeID("ClSz"), charIDToTypeID("#Pxl"), 10); return d; })()); });
t("Crystallize", function () { var d = new ActionDescriptor(); d.putInteger(charIDToTypeID("ClSz"), 10); am("crystallize", d); });
t("Facet", function () { am("facet"); });
t("Fragment", function () { am("fragment"); });
t("Clouds", function (l) { l.applyClouds(); });
t("Difference Clouds", function (l) { l.applyDifferenceClouds(); });
t("Fibers", function (l) { l.applyFibers(16, 4); });
t("Sharpen", function (l) { l.applySharpen(); });
t("Sharpen More", function (l) { l.applySharpenMore(); });
t("Sharpen Edges", function (l) { l.applySharpenEdges(); });
t("Unsharp Mask", function (l) { l.applyUnSharpMask(100, 2, 0); });
t("Emboss", function () { var d = new ActionDescriptor(); d.putInteger(charIDToTypeID("Angl"), 135); d.putInteger(charIDToTypeID("Hght"), 3); d.putInteger(charIDToTypeID("Amnt"), 100); am("emboss", d); });
t("Find Edges", function () { am("findEdges"); });
t("Solarize", function () { am("solarize"); });
t("High Pass", function (l) { l.applyHighPass(10); });
t("Maximum", function (l) { l.applyMaximum(2); });
t("Minimum", function (l) { l.applyMinimum(2); });
doc.close(SaveOptions.DONOTSAVECHANGES);
out.join("\n");

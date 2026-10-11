#target photoshop
app.displayDialogs = DialogModes.NO;
var SIDE = 4096;
var out = [];
var doc = app.documents.add(UnitValue(SIDE, "px"), UnitValue(SIDE, "px"), 72, "pf-gallery", NewDocumentMode.RGB, DocumentFill.WHITE);
doc.activeLayer.applyClouds();
doc.activeLayer.applyAddNoise(8, NoiseDistribution.UNIFORM, false);
var base = doc.historyStates.length - 1;
var ids = [["Colored Pencil","ClrP"],["Cutout","Ct  "],["Dry Brush","DryB"],["Film Grain","FlmG"],["Fresco","Frsc"],["Neon Glow","NGlw"],["Paint Daubs","PntD"],["Palette Knife","PltK"],["Plastic Wrap","PlsW"],["Poster Edges","PstE"],["Rough Pastels","RghP"],["Smudge Stick","SmdS"],["Sponge","Spng"],["Underpainting","Undr"],["Watercolor","Wtrc"],["Accented Edges","AccE"],["Angled Strokes","AngS"],["Crosshatch","Crsh"],["Dark Strokes","DrkS"],["Ink Outlines","InkO"],["Spatter","Spt "],["Sprayed Strokes","SprS"],["Sumi-e","Smie"],["Diffuse Glow","DfsG"],["Glass","Gls "],["Ocean Ripple","OcnR"],["Bas Relief","BsRl"],["Chalk & Charcoal","ChlC"],["Charcoal","Chrc"],["Chrome","Chrm"],["Conté Crayon","CntC"],["Graphic Pen","GraP"],["Halftone Pattern","HlfS"],["Note Paper","NtPr"],["Photocopy","Phtc"],["Plaster","Plst"],["Reticulation","Rtcl"],["Stamp","Stmp"],["Torn Edges","TrnE"],["Water Paper","WtrP"],["Glowing Edges","GlwE"],["Craquelure","Crql"],["Grain","Grn "],["Mosaic Tiles","MscT"],["Patchwork","Ptch"],["Stained Glass","StnG"],["Texturizer","Txtz"]];
for (var i = 0; i < ids.length; i++) {
  try {
    doc.activeHistoryState = doc.historyStates[base];
    $.hiresTimer;
    executeAction(charIDToTypeID(ids[i][1]), new ActionDescriptor(), DialogModes.NO);
    var us = $.hiresTimer;
    out.push(ids[i][0] + "\t" + (us / 1000).toFixed(1));
  } catch (e) { out.push(ids[i][0] + "\tERR " + String(e.message).split("\n")[0]); }
}
doc.close(SaveOptions.DONOTSAVECHANGES);
out.join("\n");

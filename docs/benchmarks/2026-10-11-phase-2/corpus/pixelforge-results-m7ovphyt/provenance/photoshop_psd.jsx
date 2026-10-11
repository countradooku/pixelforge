// Locally authored feature corpus. Execute only through photoshop_psd.py:
// an independent CoreGraphics process must remain able to observe dialogs.
// PF_CONFIG is data supplied by that runner, never a user document script.
#target photoshop
(function () {
    var cfg = PF_CONFIG, results = [], owned = [], current = null;
    var previousDocument = app.documents.length ? app.activeDocument : null;
    var preferences = { dialogs: app.displayDialogs, ruler: app.preferences.rulerUnits,
                        type: app.preferences.typeUnits };
    var output = new Folder(cfg.output), fatal = null, restored = false, snapshot=null;
    if(cfg.operation==="inspect_open") previousDocument=null;
    function s(k) { return stringIDToTypeID(k); }
    function c(k) { return charIDToTypeID(k); }
    function own(doc) { owned.push(doc); return doc; }
    function json(value) {
        if (value === null || value === undefined) return "null";
        if (typeof value === "string") return '"' + value.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/[\x00-\x1f]/g, function (a) { var h = a.charCodeAt(0).toString(16); return "\\u" + ("0000" + h).slice(-4); }) + '"';
        if (typeof value === "number") return isFinite(value) ? String(value) : json(String(value));
        if (typeof value === "boolean") return value ? "true" : "false";
        var out = [], i;
        if (value instanceof Array) { for (i = 0; i < value.length; i++) out.push(json(value[i])); return "[" + out.join(",") + "]"; }
        for (i in value) if (value.hasOwnProperty(i)) out.push(json(i) + ":" + json(value[i]));
        return "{" + out.join(",") + "}";
    }
    function folder(path) { var f = new Folder(path); if (!f.exists && !f.create()) throw Error("Cannot create " + path); return f; }
    function write(path, value, binary) {
        var f = new File(path); f.encoding = binary ? "BINARY" : "UTF-8";
        if (!f.open("w")) throw Error("Cannot write " + path + ": " + f.error);
        try { if (!f.write(binary ? value : json(value))) throw Error("Write failed: " + path); } finally { f.close(); }
    }
    function progress(item, stage) { write(cfg.output + "/progress.json", {id: item.id, stage: stage}); }
    function close(doc) {
        // Photoshop invalidates DOM references immediately on close; even an
        // identity comparison against a closed reference can throw.
        var ownedIndex=-1;
        for (var i=owned.length-1;i>=0;i--) if(owned[i]===doc) { ownedIndex=i; break; }
        doc.close(SaveOptions.DONOTSAVECHANGES);
        if(ownedIndex>=0) owned.splice(ownedIndex,1);
    }
    function rgb(red, green, blue) { var col = new SolidColor(); col.rgb.red = red; col.rgb.green = green; col.rgb.blue = blue; return col; }
    function rgbDescriptor(red, green, blue) {
        var d = new ActionDescriptor(); d.putDouble(c("Rd  "), red); d.putDouble(c("Grn "), green); d.putDouble(c("Bl  "), blue); return d;
    }
    function rect(doc, x, y, w, h, color, opacity) {
        doc.selection.select([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], SelectionType.REPLACE, 0, false);
        doc.selection.fill(color, ColorBlendMode.NORMAL, opacity === undefined ? 100 : opacity, false);
        doc.selection.deselect();
    }
    function ellipse(doc, x, y, w, h) {
        var d = new ActionDescriptor(), r = new ActionReference(), bounds = new ActionDescriptor();
        r.putProperty(c("Chnl"), c("fsel")); d.putReference(c("null"), r);
        bounds.putUnitDouble(c("Top "), c("#Pxl"), y); bounds.putUnitDouble(c("Left"), c("#Pxl"), x);
        bounds.putUnitDouble(c("Btom"), c("#Pxl"), y+h); bounds.putUnitDouble(c("Rght"), c("#Pxl"), x+w);
        d.putObject(c("T   "), c("Elps"), bounds); d.putBoolean(c("AntA"), true);
        executeAction(c("setd"), d, DialogModes.NO);
    }
    function targetLayer() { var r = new ActionReference(); r.putEnumerated(c("Lyr "), c("Ordn"), c("Trgt")); return r; }
    function setLayer(properties) {
        var d = new ActionDescriptor(); d.putReference(c("null"), targetLayer()); d.putObject(c("T   "), c("Lyr "), properties);
        executeAction(c("setd"), d, DialogModes.NO);
    }
    function layerDescriptor(layer) {
        var r = new ActionReference(); r.putIdentifier(c("Lyr "), layer.id); return executeActionGet(r);
    }
    function documentID() { var r=new ActionReference(); r.putProperty(c("Prpr"),s("documentID")); r.putEnumerated(c("Dcmn"),c("Ordn"),c("Trgt")); return executeActionGet(r).getInteger(s("documentID")); }
    function restoreDocument(id) { var d=new ActionDescriptor(), r=new ActionReference(); r.putIdentifier(c("Dcmn"),id); d.putReference(c("null"),r); executeAction(c("slct"),d,DialogModes.NO); }
    function base(item) {
        var w = item.params.width || 96, h = item.params.height || 72;
        var doc = own(app.documents.add(UnitValue(w,"px"), UnitValue(h,"px"), 72,
            "pf-corpus-" + item.id, NewDocumentMode.RGB, DocumentFill.TRANSPARENT, 1, BitsPerChannelType.EIGHT));
        var backdrop = doc.activeLayer; backdrop.name = "Backdrop";
        var colors = [[28,51,90],[192,64,47],[37,190,115],[215,194,49],[154,97,179],[222,222,222],[9,9,9],[126,126,126]];
        for (var i=0; i<8; i++) rect(doc, Math.floor(w*i/8),0,Math.ceil(w/8),h,rgb(colors[i][0],colors[i][1],colors[i][2]));
        var foreground = doc.artLayers.add(); foreground.name = "Foreground";
        rect(doc,w/8,h/6,w*3/4,h*2/3,rgb(211,103,71));
        rect(doc,w/3,h/3,w/3,h/3,rgb(53,123,222),63);
        doc.activeLayer = foreground;
        return doc;
    }
    var blendEnums = {normal:"NORMAL",dissolve:"DISSOLVE",darken:"DARKEN",multiply:"MULTIPLY",colorburn:"COLORBURN",linearburn:"LINEARBURN",darkercolor:"DARKERCOLOR",lighten:"LIGHTEN",screen:"SCREEN",colordodge:"COLORDODGE",lineardodge:"LINEARDODGE",lightercolor:"LIGHTERCOLOR",overlay:"OVERLAY",softlight:"SOFTLIGHT",hardlight:"HARDLIGHT",vividlight:"VIVIDLIGHT",linearlight:"LINEARLIGHT",pinlight:"PINLIGHT",hardmix:"HARDMIX",difference:"DIFFERENCE",exclusion:"EXCLUSION",subtract:"SUBTRACT",divide:"DIVIDE",hue:"HUE",saturation:"SATURATION",color:"COLORBLEND",luminosity:"LUMINOSITY"};
    function userMask(doc, feather, inverse) {
        ellipse(doc,12,10,63,49); if (feather) doc.selection.feather(feather); if (inverse) doc.selection.invert();
        var d = new ActionDescriptor(), r = new ActionReference();
        d.putClass(c("Nw  "), c("Chnl")); r.putEnumerated(c("Chnl"), c("Chnl"), c("Msk ")); d.putReference(c("At  "),r);
        d.putEnumerated(c("Usng"), c("UsrM"), c("RvlS")); executeAction(c("Mk  "),d,DialogModes.NO); doc.selection.deselect();
    }
    function makePath(doc, name, operation, ellipseShape) {
        var sub = new SubPathInfo(); sub.closed = true; sub.operation = operation || ShapeOperation.SHAPEADD;
        var coords = ellipseShape ? [[48,9],[83,36],[48,63],[13,36]] : [[13,12],[81,12],[81,59],[13,59]];
        var pts=[];
        for(var i=0;i<coords.length;i++) { var p=new PathPointInfo(); p.kind=PointKind.CORNERPOINT; p.anchor=coords[i]; p.leftDirection=coords[i]; p.rightDirection=coords[i]; pts.push(p); }
        sub.entireSubPath=pts; return doc.pathItems.add(name,[sub]);
    }
    function vectorMask(doc) {
        makePath(doc,"Vector mask construction",ShapeOperation.SHAPEADD,false);
        var select=new ActionDescriptor(), pathRef=new ActionReference(); pathRef.putName(c("Path"),"Vector mask construction"); select.putReference(c("null"),pathRef); executeAction(c("slct"),select,DialogModes.NO);
        var d=new ActionDescriptor(), r=new ActionReference(), from=new ActionReference();
        var pathClass=new ActionReference(); pathClass.putClass(c("Path")); d.putReference(c("null"),pathClass); r.putEnumerated(c("Path"),c("Path"),s("vectorMask")); d.putReference(c("At  "),r);
        from.putEnumerated(c("Path"),c("Ordn"),c("Trgt")); d.putReference(c("Usng"),from); executeAction(c("Mk  "),d,DialogModes.NO);
    }
    function gradient(level) {
        var g=new ActionDescriptor(), colors=new ActionList(), alpha=new ActionList();
        g.putString(c("Nm  "),"PixelForge authored gradient"); g.putEnumerated(c("GrdF"),c("GrdF"),c("CstS")); g.putDouble(c("Intr"),4096);
        for(var i=0;i<3;i++) { var stop=new ActionDescriptor(); stop.putObject(c("Clr "),c("RGBC"),rgbDescriptor(i===0?25:220,i===1?200:45,i===2?235:90)); stop.putEnumerated(c("Type"),c("Clry"),c("UsrS")); stop.putInteger(c("Lctn"),i*2048); stop.putInteger(c("Mdpn"),level===2?65:50); colors.putObject(c("Clrt"),stop); }
        for(var j=0;j<2;j++) { var a=new ActionDescriptor(); a.putUnitDouble(c("Opct"),c("#Prc"),j===0&&level===1?30:100); a.putInteger(c("Lctn"),j*4096); a.putInteger(c("Mdpn"),50); alpha.putObject(c("TrnS"),a); }
        g.putList(c("Clrs"),colors); g.putList(c("Trns"),alpha); return g;
    }
    function createAdjustment(kind, details) {
        var d=new ActionDescriptor(), r=new ActionReference(), using=new ActionDescriptor();
        r.putClass(c("AdjL")); d.putReference(c("null"),r); using.putObject(c("Type"),kind,details);
        d.putObject(c("Usng"),c("AdjL"),using); executeAction(c("Mk  "),d,DialogModes.NO);
    }
    function adjustment(doc,item) {
        var p=item.params.level, d=new ActionDescriptor(), kind;
        if(item.feature==="brightness") { kind=c("BrgC"); d.putInteger(c("Brgh"),[-70,25,85][p]); d.putInteger(c("Cntr"),[-30,18,70][p]); d.putBoolean(s("useLegacy"),false); }
        else if(item.feature==="exposure") { kind=s("exposure"); d.putDouble(s("exposure"),[-1.5,0.7,2][p]); d.putDouble(s("offset"),[0,0.02,-0.02][p]); d.putDouble(s("gammaCorrection"),[1,1.2,0.8][p]); }
        else if(item.feature==="vibrance") { kind=s("vibrance"); d.putInteger(s("vibrance"),[-60,30,80][p]); d.putInteger(c("Strt"),[-20,15,35][p]); }
        else if(item.feature==="invert") kind=c("Invr");
        else if(item.feature==="posterize") { kind=c("Pstr"); d.putInteger(c("Lvls"),[2,5,13][p]); }
        else if(item.feature==="threshold") { kind=c("Thrs"); d.putInteger(c("Lvl "),[45,128,208][p]); }
        else if(item.feature==="levels") {
            kind=c("Lvls"); var ls=new ActionList(), lv=new ActionDescriptor(), ref=new ActionReference(), input=new ActionList(), out=new ActionList();
            ref.putEnumerated(c("Chnl"),c("Chnl"),p===2?c("Rd  "):c("Cmps")); lv.putReference(c("Chnl"),ref);
            input.putInteger([8,25,40][p]); input.putInteger([245,225,210][p]); out.putInteger(3); out.putInteger(248);
            lv.putList(c("Inpt"),input); lv.putDouble(c("Gmm "),[0.75,1.3,1.65][p]); lv.putList(c("Otpt"),out); ls.putObject(c("LvlA"),lv); d.putList(c("Adjs"),ls);
        } else if(item.feature==="curves") {
            kind=c("Crvs"); var curves=new ActionList(), curve=new ActionDescriptor(), cr=new ActionReference(), pts=new ActionList();
            cr.putEnumerated(c("Chnl"),c("Chnl"),p===2?c("Bl  "):c("Cmps")); curve.putReference(c("Chnl"),cr);
            var points=[[0,0],[64,[30,90,45][p]],[128,[160,100,190][p]],[255,255]];
            for(var q=0;q<points.length;q++) { var pt=new ActionDescriptor(); pt.putDouble(c("Hrzn"),points[q][0]); pt.putDouble(c("Vrtc"),points[q][1]); pts.putObject(c("Pnt "),pt); }
            curve.putList(c("Crv "),pts); curves.putObject(c("CrvA"),curve); d.putList(c("Adjs"),curves);
        } else if(item.feature==="huesaturation") {
            kind=c("HStr"); var hs=new ActionList(), hv=new ActionDescriptor(); hv.putInteger(c("H   "),[-55,25,110][p]); hv.putInteger(c("Strt"),[-35,20,45][p]); hv.putInteger(c("Lght"),[-8,4,12][p]); hs.putObject(c("Hst2"),hv); d.putList(c("Adjs"),hs); d.putBoolean(c("Clrz"),p===2);
        } else if(item.feature==="colorbalance") {
            kind=c("ClrB"); var keys=["ShdL","MdtL","HghL"];
            for(var b=0;b<3;b++) { var values=new ActionList(); values.putInteger(b===p?35:-5); values.putInteger(b===p?-20:0); values.putInteger(b===p?15:0); d.putList(c(keys[b]),values); } d.putBoolean(c("PrsL"),p!==2);
        } else if(item.feature==="blackwhite") {
            kind=s("blackAndWhite"); var channels=["Rd  ","Yllw","Grn ","Cyn ","Bl  ","Mgnt"];
            for(var bw=0;bw<6;bw++) d.putInteger(c(channels[bw]),20+bw*12+p*15); d.putBoolean(s("useTint"),p===2); d.putObject(s("tintColor"),c("RGBC"),rgbDescriptor(140,90,35));
        } else if(item.feature==="photofilter") { kind=s("photoFilter"); d.putObject(c("Clr "),c("RGBC"),rgbDescriptor([220,40,190][p],[100,140,55][p],[35,230,160][p])); d.putInteger(s("density"),[15,45,80][p]); d.putBoolean(c("PrsL"),p!==2); }
        else if(item.feature==="channelmixer") {
            kind=c("ChnM"); d.putBoolean(c("Mnch"),p===2);
            for(var mc=0;mc<(p===2?1:3);mc++) { var md=new ActionDescriptor(); md.putUnitDouble(c("Rd  "),c("#Prc"),mc===0?110:10); md.putUnitDouble(c("Grn "),c("#Prc"),mc===1?80:20); md.putUnitDouble(c("Bl  "),c("#Prc"),mc===2?95:-5); md.putUnitDouble(c("Cnst"),c("#Prc"),p*3); d.putObject(p===2?c("Gry "):c(["Rd  ","Grn ","Bl  "][mc]),s("channelMatrix"),md); }
        } else if(item.feature==="selectivecolor") {
            kind=c("SlcC"); d.putEnumerated(c("Mthd"),c("Mthd"),p===2?c("Absl"):c("Rltv")); var sc=new ActionList(), sd=new ActionDescriptor(); sd.putEnumerated(s("colors"),s("colors"),s(["reds","neutrals","blacks"][p])); sd.putUnitDouble(c("Cyn "),c("#Prc"),20); sd.putUnitDouble(c("Mgnt"),c("#Prc"),-30); sd.putUnitDouble(c("Ylw "),c("#Prc"),10); sd.putUnitDouble(c("Blck"),c("#Prc"),5); sc.putObject(s("colorCorrection"),sd); d.putList(s("colorCorrection"),sc);
        } else if(item.feature==="gradientmap") { kind=c("GdMp"); d.putObject(c("Grad"),c("Grdn"),gradient(p)); d.putBoolean(c("Rvrs"),p===1); d.putBoolean(c("Dthr"),p===2); }
        else if(item.feature==="colorlookup") throw Error("Color Lookup fixture requires an authored LUT payload; no downloaded or preset-dependent substitute is used");
        else throw Error("No adjustment generator: "+item.feature);
        if(item.feature==="selectivecolor") { var initial=new ActionDescriptor(); initial.putEnumerated(s("presetKind"),s("presetKindType"),s("presetKindDefault")); createAdjustment(kind,initial); d.putEnumerated(s("presetKind"),s("presetKindType"),s("presetKindCustom")); var correctionSet=new ActionDescriptor(), correctionRef=new ActionReference(); correctionRef.putEnumerated(c("AdjL"),c("Ordn"),c("Trgt")); correctionSet.putReference(c("null"),correctionRef); correctionSet.putObject(c("T   "),kind,d); executeAction(c("setd"),correctionSet,DialogModes.NO); doc.activeLayer.name="Editable "+item.feature; return; }
        createAdjustment(kind,d); doc.activeLayer.name="Editable "+item.feature;
    }
    function createFill(kind,details) {
        var d=new ActionDescriptor(), r=new ActionReference(), layer=new ActionDescriptor(); r.putClass(s("contentLayer")); d.putReference(c("null"),r);
        layer.putObject(c("Type"),kind,details); d.putObject(c("Usng"),s("contentLayer"),layer); executeAction(c("Mk  "),d,DialogModes.NO);
    }
    function fill(doc,item) {
        var d=new ActionDescriptor(), n=item.params.level;
        if(item.feature==="solid") { d.putObject(c("Clr "),c("RGBC"),rgbDescriptor([12,197,72][n],[90,34,185][n],[223,104,44][n])); createFill(s("solidColorLayer"),d); }
        else if(item.feature==="gradient") { d.putObject(c("Grad"),c("Grdn"),gradient(n)); d.putUnitDouble(c("Angl"),c("#Ang"),[0,45,120][n]); d.putEnumerated(c("Type"),c("GrdT"),c(["Lnr ","Rdl ","Angl"][n])); d.putUnitDouble(c("Scl "),c("#Prc"),[70,110,160][n]); d.putBoolean(c("Dthr"),true); createFill(s("gradientLayer"),d); }
        else throw Error("Pattern fill needs a locally authored pattern resource and verified preset cleanup; not substituted by a flat fill");
        doc.activeLayer.name="Editable "+item.feature+" fill";
    }
    function style(doc,item) {
        if(item.variant==="contour"&&(item.feature==="solidFill"||item.feature==="gradientFill"||item.feature==="patternFill"||item.feature==="frameFX")) throw Error("Invalid feature combination: Photoshop does not expose a contour for "+item.feature+"; historical duplicate is not coverage");
        if(item.feature==="patternFill") throw Error("Pattern overlay authoring needs a locally authored pattern resource; no built-in preset assumption");
        var effects=new ActionDescriptor(), e=new ActionDescriptor(), large=item.variant==="large", effect=item.feature;
        effects.putUnitDouble(s("scale"),c("#Prc"),large?150:100);
        e.putBoolean(s("enabled"),true); e.putBoolean(s("present"),true); e.putBoolean(s("showInDialog"),true);
        e.putEnumerated(s("mode"),s("blendMode"),s(item.variant==="blend"?"screen":effect==="solidFill"?"normal":"multiply"));
        e.putObject(s("color"),s("RGBColor"),rgbDescriptor(56,92,180)); e.putUnitDouble(s("opacity"),s("percentUnit"),large?83:47);
        e.putBoolean(s("useGlobalAngle"),true); e.putUnitDouble(s("localLightingAngle"),s("angleUnit"),120);
        e.putUnitDouble(s("distance"),s("pixelsUnit"),large?14:4); e.putUnitDouble(s("chokeMatte"),s("pixelsUnit"),large?5:0);
        e.putUnitDouble(s("blur"),s("pixelsUnit"),large?12:3); e.putUnitDouble(s("noise"),s("percentUnit"),0);
        e.putBoolean(s("antiAlias"),true); e.putBoolean(s("layerConceals"),true);
        if(effect==="bevelEmboss") {
            e.putEnumerated(s("bevelStyle"),s("bevelStyle"),s(large?"outerBevel":"innerBevel")); e.putEnumerated(s("bevelTechnique"),s("bevelTechnique"),s("softMatte"));
            e.putUnitDouble(s("strengthRatio"),s("percentUnit"),large?250:100); e.putEnumerated(s("bevelDirection"),s("bevelDirection"),s("in"));
            e.putUnitDouble(s("softness"),s("pixelsUnit"),1); e.putUnitDouble(s("localLightingAltitude"),s("angleUnit"),30);
            e.putEnumerated(s("highlightMode"),s("blendMode"),s("screen")); e.putObject(s("highlightColor"),s("RGBColor"),rgbDescriptor(255,255,255)); e.putUnitDouble(s("highlightOpacity"),s("percentUnit"),75);
            e.putEnumerated(s("shadowMode"),s("blendMode"),s("multiply")); e.putObject(s("shadowColor"),s("RGBColor"),rgbDescriptor(0,0,0)); e.putUnitDouble(s("shadowOpacity"),s("percentUnit"),75);
        }
        if(effect==="gradientFill") { e.putObject(s("gradient"),s("gradientClassEvent"),gradient(large?2:0)); e.putUnitDouble(s("angle"),s("angleUnit"),45); e.putEnumerated(s("type"),s("gradientType"),s(large?"radial":"linear")); e.putUnitDouble(s("scale"),s("percentUnit"),100); e.putBoolean(s("align"),true); }
        if(effect==="frameFX") { e.putEnumerated(s("style"),s("frameStyle"),s(large?"insetFrame":"outsetFrame")); e.putEnumerated(s("paintType"),s("frameFill"),s("solidColor")); e.putUnitDouble(s("size"),s("pixelsUnit"),large?10:3); }
        if(effect==="innerGlow") e.putEnumerated(s("innerGlowSource"),s("innerGlowSource"),s(large?"centerGlow":"edgeGlow"));
        if(effect==="innerGlow"||effect==="outerGlow") { e.putEnumerated(s("glowTechnique"),s("glowTechnique"),s("softMatte")); e.putUnitDouble(s("inputRange"),s("percentUnit"),50); }
        if(item.variant==="contour") {
            var contour=new ActionDescriptor(), curve=new ActionList(); contour.putString(s("name"),"Authored valley");
            for(var i=0;i<3;i++) { var cp=new ActionDescriptor(); cp.putDouble(s("horizontal"),i*127.5); cp.putDouble(s("vertical"),i===1?25:i*127.5); curve.putObject(s("curvePoint"),cp); }
            contour.putList(s("curve"),curve); e.putObject(s(effect==="chromeFX"?"mappingShape":"transferSpec"),s("shapeCurveType"),contour);
        }
        if(item.variant==="multiple") {
            if(effect!=="dropShadow"&&effect!=="innerShadow"&&effect!=="solidFill"&&effect!=="gradientFill"&&effect!=="frameFX") throw Error("Photoshop does not support multiple instances for "+effect+"; case must be classified as unavailable, not passed");
            var multi=new ActionList(); multi.putObject(s(effect),e); var second=new ActionDescriptor(); second.fromStream(e.toStream()); second.putObject(s("color"),s("RGBColor"),rgbDescriptor(213,42,90)); second.putUnitDouble(s("distance"),s("pixelsUnit"),9); second.putUnitDouble(s("size"),s("pixelsUnit"),6); multi.putObject(s(effect),second); effects.putList(s(effect+"Multi"),multi);
        } else effects.putObject(s(effect),s(effect),e);
        var set=new ActionDescriptor(), ref=new ActionReference(); ref.putProperty(c("Prpr"),s("layerEffects")); ref.putEnumerated(c("Lyr "),c("Ordn"),c("Trgt")); set.putReference(c("null"),ref); set.putObject(c("T   "),s("layerEffects"),effects); executeAction(c("setd"),set,DialogModes.NO);
    }
    function text(doc,item) {
        if(item.feature==="area") throw Error("Arbitrary area-text authoring descriptor is not yet implemented; no rectangular paragraph substitute");
        var layer=doc.artLayers.add(); layer.kind=LayerKind.TEXT; layer.name="Live text "+item.feature;
        var ti=layer.textItem; ti.font="Helvetica"; ti.size=11; ti.color=rgb(22,37,63); ti.position=[8,25];
        ti.contents=item.feature==="unicode"?"A\u00e9\u03a9\ud83d\ude00\r\u6f22\u5b57": "The quick fox\rjumps again";
        if(item.feature==="paragraph"||item.feature==="paragraph-styles") { ti.kind=TextType.PARAGRAPHTEXT; ti.position=[8,8]; ti.width=73; ti.height=56; ti.contents="A short paragraph with wrapping words and a second line.\rLast paragraph."; ti.justification=Justification.LEFT; ti.hyphenation=true; }
        if(item.feature==="paragraph-styles") { ti.firstLineIndent=5; ti.leftIndent=3; ti.rightIndent=4; ti.spaceBefore=2; ti.spaceAfter=3; ti.justification=Justification.LEFTJUSTIFIED; }
        if(item.feature==="vertical") { ti.direction=Direction.VERTICAL; ti.contents="Vertical"; ti.position=[45,8]; }
        if(item.feature==="warp") { ti.warpStyle=WarpStyle[item.params.warp]; ti.warpBend=item.params.bend; ti.warpHorizontalDistortion=20; ti.warpVerticalDistortion=-15; ti.warpDirection=Direction.HORIZONTAL; }
        if(item.feature==="path") {
            ti.contents="Text along a curve"; ti.size=9;
            var textPath=new SubPathInfo(); textPath.closed=false; textPath.operation=ShapeOperation.SHAPEADD;
            var curvePoints=[], geometry=[[[10,44],[10,44],[22,14]],[[48,25],[33,25],[63,25]],[[86,44],[74,14],[86,44]]];
            for(var pi=0;pi<geometry.length;pi++) { var point=new PathPointInfo(); point.kind=PointKind.SMOOTHPOINT; point.anchor=geometry[pi][0]; point.leftDirection=geometry[pi][1]; point.rightDirection=geometry[pi][2]; curvePoints.push(point); }
            textPath.entireSubPath=curvePoints; doc.pathItems.add("Text curve construction",[textPath]);
            var selectPath=new ActionDescriptor(), pathReference=new ActionReference(); pathReference.putName(c("Path"),"Text curve construction"); selectPath.putReference(c("null"),pathReference); executeAction(c("slct"),selectPath,DialogModes.NO);
            var pathQuery=new ActionReference(); pathQuery.putProperty(c("Prpr"),s("pathContents")); pathQuery.putEnumerated(c("Path"),c("Ordn"),c("Trgt"));
            var pathData=executeActionGet(pathQuery).getObjectValue(s("pathContents"));
            var textData=layerDescriptor(layer).getObjectValue(s("textKey")), shapes=textData.getList(s("textShape")), textShape=shapes.getObjectValue(0), range=new ActionDescriptor(), origin=new ActionDescriptor();
            textShape.putObject(s("path"),s("pathClass"),pathData); textShape.putEnumerated(s("char"),s("char"),s("onACurve")); textShape.putInteger(s("pathTypeSpacing"),0);
            range.putDouble(s("start"),0); range.putDouble(s("end"),2); textShape.putObject(s("tRange"),s("range"),range);
            var newShapes=new ActionList(); newShapes.putObject(s("textShape"),textShape); textData.putList(s("textShape"),newShapes);
            origin.putUnitDouble(s("horizontal"),c("#Prc"),0); origin.putUnitDouble(s("vertical"),c("#Prc"),0); textData.putObject(s("textClickPoint"),c("Pnt "),origin);
            var textSet=new ActionDescriptor(), textReference=new ActionReference(); textReference.putEnumerated(c("TxLr"),c("Ordn"),c("Trgt")); textSet.putReference(c("null"),textReference); textSet.putObject(c("T   "),c("TxLr"),textData); executeAction(c("setd"),textSet,DialogModes.NO);
        }
        if(item.feature==="runs") {
            var d=new ActionDescriptor(), r=new ActionReference(), tx=new ActionDescriptor(), list=new ActionList();
            r.putEnumerated(c("TxLr"),c("Ordn"),c("Trgt")); d.putReference(c("null"),r);
            var ranges=[[0,4,14,200,25,60],[4,ti.contents.length,10,25,110,210]];
            for(var i=0;i<ranges.length;i++) { var range=new ActionDescriptor(), st=new ActionDescriptor(), v=ranges[i]; range.putInteger(c("From"),v[0]); range.putInteger(c("T   "),v[1]); st.putString(s("fontPostScriptName"),"Helvetica"); st.putUnitDouble(c("Sz  "),c("#Pnt"),v[2]); st.putObject(c("Clr "),c("RGBC"),rgbDescriptor(v[3],v[4],v[5])); st.putInteger(s("tracking"),i*80); range.putObject(c("TxtS"),c("TxtS"),st); list.putObject(c("Txtt"),range); }
            tx.putList(c("Txtt"),list); d.putObject(c("T   "),c("TxLr"),tx); executeAction(c("setd"),d,DialogModes.NO);
        }
    }
    function shape(doc,item) {
        if(item.feature==="paths-panel"||item.feature==="clipping-path") { var path=makePath(doc,"Authored path",ShapeOperation.SHAPEADD,false); if(item.feature==="clipping-path") path.makeClippingPath(0.5); return; }
        if(item.feature==="pattern-fill") throw Error("Pattern shape fill requires a locally authored pattern resource");
        var kind=item.feature, d=new ActionDescriptor(), r=new ActionReference(), layer=new ActionDescriptor(), fill=new ActionDescriptor(), geo=new ActionDescriptor();
        r.putClass(s("contentLayer")); d.putReference(c("null"),r); fill.putObject(c("Clr "),c("RGBC"),rgbDescriptor(32,174,129));
        if(kind==="gradient-fill") { makePath(doc,"Gradient shape path",ShapeOperation.SHAPEADD,false); var gradientShape=new ActionDescriptor(); gradientShape.putObject(c("Grad"),c("Grdn"),gradient(1)); gradientShape.putUnitDouble(c("Angl"),c("#Ang"),35); gradientShape.putEnumerated(c("Type"),c("GrdT"),c("Lnr ")); createFill(s("gradientLayer"),gradientShape); return; }
        if(item.params.operation&&(item.params.operation!=="add"||kind==="ellipse"||kind==="rounded")) {
            function component(geometry,operation,second) {
                var sub=new SubPathInfo(); sub.closed=true; sub.operation=operation; var points=[], data;
                if(second) data=[[43,28],[89,28],[89,64],[43,64]];
                else if(geometry==="ellipse") data=[[48,13],[82,36],[48,59],[14,36]];
                else if(geometry==="polygon") data=[[48,9],[85,31],[71,61],[21,61],[11,31]];
                else if(geometry==="rounded") data=[[23,13],[73,13],[82,22],[82,50],[73,59],[23,59],[14,50],[14,22]];
                else data=[[14,13],[82,13],[82,59],[14,59]];
                for(var i=0;i<data.length;i++) { var point=new PathPointInfo(); point.kind=PointKind.CORNERPOINT; point.anchor=data[i]; point.leftDirection=data[i]; point.rightDirection=data[i];
                    if(!second&&geometry==="ellipse") { var handles=[[[29.2223,13],[66.7777,13]],[[82,23.29745],[82,48.70255]],[[66.7777,59],[29.2223,59]],[[14,48.70255],[14,23.29745]]]; point.kind=PointKind.SMOOTHPOINT; point.leftDirection=handles[i][0]; point.rightDirection=handles[i][1]; }
                    if(!second&&geometry==="rounded") { var rounded=[[[18.029437,13],[23,13]],[[73,13],[77.970563,13]],[[82,17.029437],[82,22]],[[82,50],[82,54.970563]],[[77.970563,59],[73,59]],[[23,59],[18.029437,59]],[[14,54.970563],[14,50]],[[14,22],[14,17.029437]]]; point.leftDirection=rounded[i][0]; point.rightDirection=rounded[i][1]; }
                    points.push(point);
                } sub.entireSubPath=points; return sub;
            }
            var operations={subtract:ShapeOperation.SHAPESUBTRACT,intersect:ShapeOperation.SHAPEINTERSECT,xor:ShapeOperation.SHAPEXOR};
            var components=[component(kind,ShapeOperation.SHAPEADD,false)]; if(item.params.operation!=="add") components.push(component("rectangle",operations[item.params.operation],true));
            doc.pathItems.add("Authored "+kind+" "+item.params.operation,components);
            createFill(s("solidColorLayer"),fill); doc.activeLayer.name="Compound "+kind+" "+item.params.operation; return;
        }
        layer.putObject(c("Type"),s("solidColorLayer"),fill); geo.putUnitDouble(c("Top "),c("#Pxl"),12); geo.putUnitDouble(c("Left"),c("#Pxl"),15); geo.putUnitDouble(c("Btom"),c("#Pxl"),58); geo.putUnitDouble(c("Rght"),c("#Pxl"),80);
        if(kind==="rounded") geo.putUnitDouble(c("Rds "),c("#Pxl"),9);
        if(kind==="polygon") { makePath(doc,"Polygon construction",ShapeOperation.SHAPEADD,true); createFill(s("solidColorLayer"),fill); }
        else { layer.putObject(c("Shp "),kind==="ellipse"?s("ellipse"):kind==="rounded"?s("roundedRectangle"):c("Rctn"),geo); d.putObject(c("Usng"),s("contentLayer"),layer); executeAction(c("Mk  "),d,DialogModes.NO); }
        if(kind==="stroke-solid"||kind==="stroke-dashed") {
            var stroke=new ActionDescriptor(), content=new ActionDescriptor(); stroke.putInteger(s("strokeStyleVersion"),2); stroke.putBoolean(s("strokeEnabled"),true); stroke.putBoolean(s("fillEnabled"),true); stroke.putUnitDouble(s("strokeStyleLineWidth"),c("#Pxl"),4); stroke.putEnumerated(s("strokeStyleLineAlignment"),s("strokeStyleLineAlignment"),s("strokeStyleAlignInside")); content.putObject(c("Clr "),c("RGBC"),rgbDescriptor(210,35,80)); stroke.putObject(s("strokeStyleContent"),s("solidColorLayer"),content);
            if(kind==="stroke-dashed") { var dash=new ActionList(); dash.putUnitDouble(c("#Pnt"),3); dash.putUnitDouble(c("#Pnt"),2); stroke.putList(s("strokeStyleLineDashSet"),dash); }
            var props=new ActionDescriptor(), strokeSet=new ActionDescriptor(), strokeRef=new ActionReference(); props.putObject(s("strokeStyle"),s("strokeStyle"),stroke); strokeRef.putEnumerated(s("contentLayer"),c("Ordn"),c("Trgt")); strokeSet.putReference(c("null"),strokeRef); strokeSet.putObject(c("T   "),s("shapeStyle"),props); executeAction(c("setd"),strokeSet,DialogModes.NO);
        }
    }
    function smart(doc,item) {
        var assets=folder(cfg.output+"/assets"), path=assets.fsName+"/"+item.id+(item.feature==="embedded-psd"?".psd":".png");
        if(item.feature==="embedded-psd") { var source=own(doc.duplicate("pf-export-smart-source",false)); try { var extra=source.artLayers.add(); extra.name="Embedded document second layer"; rect(source,4,4,16,12,rgb(246,196,53)); var options=new PhotoshopSaveOptions(); options.layers=true; options.embedColorProfile=true; options.maximizeCompatibility=true; source.saveAs(new File(path),options,true,Extension.LOWERCASE); } finally { close(source); app.activeDocument=doc; } }
        else savePng(doc,path);
        var place=new ActionDescriptor(); place.putPath(c("null"),new File(path)); place.putEnumerated(c("FTcs"),c("QCSt"),c("Qcsa")); place.putBoolean(s("linked"),item.feature==="linked");
        executeAction(c("Plc "),place,DialogModes.NO); doc.activeLayer.name="Smart "+item.feature;
        if(item.feature==="transform") { doc.activeLayer.resize(78,113,AnchorPosition.MIDDLECENTER); doc.activeLayer.rotate(17,AnchorPosition.MIDDLECENTER); }
        if(item.feature==="warp") { var transform=new ActionDescriptor(), target=new ActionReference(), warp=layerDescriptor(doc.activeLayer).getObjectValue(s("smartObjectMore")).getObjectValue(s("warp")); target.putEnumerated(c("Lyr "),c("Ordn"),c("Trgt")); transform.putReference(c("null"),target); transform.putEnumerated(c("FTcs"),c("QCSt"),c("Qcsa")); warp.putEnumerated(s("warpStyle"),s("warpStyle"),s("warpArc")); warp.putDouble(s("warpValue"),23); warp.putDouble(s("warpPerspective"),7); warp.putDouble(s("warpPerspectiveOther"),-5); transform.putObject(s("warp"),s("warp"),warp); executeAction(c("Trnf"),transform,DialogModes.NO); }
        if(item.feature==="smart-filter") doc.activeLayer.applyGaussianBlur(3.5);
    }
    function mode(doc,item) {
        if(item.feature==="rgb") return;
        doc.flatten();
        if(item.feature==="duotone") { doc.changeMode(ChangeMode.GRAYSCALE); var convert=new ActionDescriptor(), duotone=new ActionDescriptor(), inks=new ActionList(); for(var inkIndex=0;inkIndex<2;inkIndex++) { var ink=new ActionDescriptor(), response=new ActionList(); ink.putString(c("Nm  "),inkIndex===0?"Authored black":"Authored violet"); ink.putObject(c("Clr "),c("RGBC"),inkIndex===0?rgbDescriptor(0,0,0):rgbDescriptor(135,39,179)); for(var pi=0;pi<3;pi++) { var cp=new ActionDescriptor(); cp.putInteger(c("Vl  "),pi*50); cp.putDouble(c("Rspn"),pi===1?(inkIndex===0?35:62):pi*50); response.putObject(c("DtnP"),cp); } ink.putList(c("Crv "),response); inks.putObject(c("DtnI"),ink); } duotone.putList(c("Inks"),inks); convert.putObject(c("T   "),c("DtnM"),duotone); executeAction(c("CnvM"),convert,DialogModes.NO); }
        else if(item.feature==="bitmap") { doc.changeMode(ChangeMode.GRAYSCALE); var options=new BitmapConversionOptions(); options.method=BitmapConversionType.DIFFUSIONDITHER; options.resolution=72; doc.changeMode(ChangeMode.BITMAP,options); }
        else if(item.feature==="indexed") { var indexed=new IndexedConversionOptions(); indexed.palette=Palette.LOCALPERCEPTUAL; indexed.colors=32; indexed.dither=Dither.NONE; doc.changeMode(ChangeMode.INDEXEDCOLOR,indexed); }
        else doc.changeMode({grayscale:ChangeMode.GRAYSCALE,lab:ChangeMode.LAB,cmyk:ChangeMode.CMYK,multichannel:ChangeMode.MULTICHANNEL}[item.feature]);
    }
    function metadata(doc,item) {
        if(item.feature==="guides") { doc.guides.add(Direction.HORIZONTAL,UnitValue(19,"px")); doc.guides.add(Direction.VERTICAL,UnitValue(31,"px")); }
        else if(item.feature==="xmp-iptc") { doc.info.author="PixelForge generated fixture"; doc.info.caption="Local test, no Adobe sample assets"; doc.info.keywords=["PSD corpus","Unicode \u03a9"]; doc.info.copyrightNotice="Generated for local interoperability testing"; }
        else if(item.feature==="icc-resolution") { doc.resizeImage(undefined,undefined,144,ResampleMethod.NONE); doc.convertProfile("sRGB IEC61966-2.1",Intent.RELATIVECOLORIMETRIC,true,false); }
        else if(item.feature==="layer-comps") { doc.layerComps.add("Full","All visible",true,true,true); doc.activeLayer.visible=false; doc.layerComps.add("Backdrop only","Foreground hidden",true,true,true); doc.activeLayer.visible=true; }
        else if(item.feature==="print") { doc.printSettings.flip=false; doc.printSettings.negative=false; doc.printSettings.caption=true; }
        else if(item.feature==="timeline") executeAction(s("makeFrameAnimation"),new ActionDescriptor(),DialogModes.NO);
        else if(item.feature==="count") { doc.countItems.add([UnitValue(21,"px"),UnitValue(17,"px")]); doc.countItems.add([UnitValue(48,"px"),UnitValue(33,"px")]); doc.countItems.add([UnitValue(75,"px"),UnitValue(51,"px")]); }
        else if(item.feature==="artboard") { var artboard=new ActionDescriptor(), artboardRef=new ActionReference(), artboardInfo=new ActionDescriptor(), artboardBounds=new ActionDescriptor(); artboardRef.putClass(s("artboardSection")); artboard.putReference(c("null"),artboardRef); artboardInfo.putString(c("Nm  "),"Authored artboard"); artboard.putObject(c("Usng"),s("artboardSection"),artboardInfo); artboardBounds.putDouble(c("Top "),0); artboardBounds.putDouble(c("Left"),0); artboardBounds.putDouble(c("Btom"),72); artboardBounds.putDouble(c("Rght"),96); artboard.putObject(s("artboardRect"),s("classFloatRect"),artboardBounds); executeAction(c("Mk  "),artboard,DialogModes.NO); }
        else throw Error("Authoring for "+item.feature+" is not implemented; feature remains an acceptance failure");
    }
    function author(doc,item) {
        var layer=doc.activeLayer, p=item.params;
        if(item.family==="blend") { layer.blendMode=BlendMode[blendEnums[item.feature]]; layer.opacity=p.opacity; layer.fillOpacity=p.fill; }
        else if(item.family==="pixels") {
            if(item.feature==="transparent") doc.layers[1].visible=false;
            else if(item.feature==="partial-alpha") { doc.layers[1].visible=false; layer.opacity=41; }
            else if(item.feature==="negative-offset") layer.translate(-24,-17);
            else if(item.feature==="offcanvas") layer.translate(54,22);
            else if(item.feature==="unicode-name") layer.name="\u03a9 \u65e5\u672c\u8a9e \ud83d\ude00 / layer";
            else if(item.feature==="empty-layer") { doc.artLayers.add().name="Empty pixels"; }
        } else if(item.family==="mask") {
            if(item.feature==="vector"||item.feature==="user-and-vector") vectorMask(doc);
            if(item.feature!=="vector") userMask(doc,p.feather||0,p.inverse||false);
            if(item.feature==="disabled") { var maskProp=new ActionDescriptor(); maskProp.putBoolean(s("userMaskEnabled"),false); setLayer(maskProp); }
        } else if(item.family==="group") {
            var group=doc.layerSets.add(); group.name="Group "+item.feature; layer.move(group,ElementPlacement.INSIDE); group.opacity=p.opacity;
            group.blendMode=item.feature==="passthrough"?BlendMode.PASSTHROUGH:BlendMode.MULTIPLY;
            if(item.feature==="nested") { var inner=group.layerSets.add(); inner.name="Nested group"; layer.move(inner,ElementPlacement.INSIDE); inner.blendMode=BlendMode.SCREEN; }
        } else if(item.family==="layer") {
            if(item.feature==="clipping") layer.grouped=true;
            else if(item.feature==="hidden") layer.visible=false;
            else if(item.feature==="lock-all") layer.allLocked=true;
            else if(item.feature==="lock-position") layer.positionLocked=true;
            else if(item.feature==="lock-transparency") layer.transparentPixelsLocked=true;
            else { var prop=new ActionDescriptor();
                if(item.feature==="color-tag") prop.putEnumerated(c("Clr "),c("Clr "),c("Orng"));
                else if(item.feature==="knockout-shallow"||item.feature==="knockout-deep") prop.putEnumerated(s("knockout"),s("knockout"),s(item.feature==="knockout-deep"?"deep":"shallow"));
                else if(item.feature==="channels") { var restrictions=new ActionList(); restrictions.putEnumerated(c("Chnl"),c("Rd  ")); prop.putList(s("channelRestrictions"),restrictions); }
                else if(item.feature==="blend-if") { var ranges=new ActionList(), range=new ActionDescriptor(), rr=new ActionReference(); rr.putEnumerated(c("Chnl"),c("Chnl"),c("Gry ")); range.putReference(c("Chnl"),rr); range.putInteger(c("SrcB"),20); range.putInteger(c("Srcl"),65); range.putInteger(c("SrcW"),230); range.putInteger(c("Srcm"),190); range.putInteger(c("DstB"),0); range.putInteger(c("Dstl"),0); range.putInteger(c("DstW"),255); range.putInteger(c("Dstt"),255); ranges.putObject(c("Blnd"),range); prop.putList(c("Blnd"),ranges); }
                setLayer(prop);
            }
        } else if(item.family==="adjustment") adjustment(doc,item);
        else if(item.family==="fill") fill(doc,item);
        else if(item.family==="style") style(doc,item);
        else if(item.family==="text") text(doc,item);
        else if(item.family==="shape") shape(doc,item);
        else if(item.family==="smart") smart(doc,item);
        else if(item.family==="mode") mode(doc,item);
        else if(item.family==="channel") {
            var count=item.feature.indexOf("multiple")>=0?3:1;
            for(var ch=0;ch<count;ch++) { var channel=doc.channels.add(); channel.name=(item.feature.indexOf("spot")===0?"Spot ":"Alpha ")+ch; channel.kind=item.feature.indexOf("spot")===0?ChannelType.SPOTCOLOR:ChannelType.MASKEDAREA; channel.color=rgb(210,40+ch*45,110); channel.opacity=35+ch*20; doc.activeChannels=[channel]; rect(doc,12+ch*5,9+ch*5,40,35,rgb(255,255,255)); } doc.activeChannels=doc.componentChannels;
        } else if(item.family==="metadata") metadata(doc,item);
        else if(item.family==="large") {
            for(var i=0;i<4;i++) { var l=doc.artLayers.add(); l.name="Large edit layer "+i; rect(doc,p.width*(i+1)/8,p.height*(i+1)/10,p.width/4,p.height/4,rgb(40+i*40,175-i*20,80+i*30)); l.opacity=35+i*15; }
            doc.guides.add(Direction.VERTICAL,UnitValue(p.width/2,"px"));
        } else throw Error("Unknown feature family "+item.family);
    }
    function verifyAuthored(doc,item) {
        // A command that silently ignores an unknown property must not turn
        // into claimed feature coverage merely because the file saves.
        var layer=doc.activeLayer, d=layerDescriptor(layer), key;
        if(item.family==="adjustment"||item.family==="fill") {
            if(layer.typename!=="ArtLayer"||layer.kind===LayerKind.NORMAL||!d.hasKey(s("adjustment"))) throw Error("Authoring verification: editable adjustment/fill was not created");
            var authoredAdjustment=d.getList(s("adjustment")).getObjectValue(0);
            if(item.feature==="channelmixer") { var matrix=authoredAdjustment.getObjectValue(item.params.level===2?c("Gry "):c("Rd  ")); if(matrix.getUnitDoubleValue(c("Rd  "))!==110||matrix.getUnitDoubleValue(c("Cnst"))!==item.params.level*3) throw Error("Authoring verification: requested channel mixer coefficients were ignored"); }
            if(item.feature==="selectivecolor"&&(!authoredAdjustment.hasKey(s("colorCorrection"))||authoredAdjustment.getList(s("colorCorrection")).count===0)) throw Error("Authoring verification: requested Selective Color corrections were ignored");
        }
        if(item.family==="text") {
            if(layer.kind!==LayerKind.TEXT) throw Error("Authoring verification: text is not live");
            if(item.feature==="warp"&&layer.textItem.warpStyle!==WarpStyle[item.params.warp]) throw Error("Authoring verification: requested text warp is absent");
            if(item.feature==="path") { var textShapes=d.getObjectValue(s("textKey")).getList(s("textShape")), authoredShape=textShapes.getObjectValue(0); if(!authoredShape.hasKey(s("path"))||authoredShape.getEnumerationValue(s("char"))!==s("onACurve")) throw Error("Authoring verification: text path is absent"); }
        }
        if(item.family==="smart") {
            if(!d.hasKey(s("smartObject"))) throw Error("Authoring verification: placed smart object is absent");
            var smart=d.getObjectValue(s("smartObject"));
            if(item.feature==="linked"&&(!smart.hasKey(s("linked"))||!smart.getBoolean(s("linked")))) throw Error("Authoring verification: smart object was embedded instead of linked");
            if(item.feature==="warp"&&d.getObjectValue(s("smartObjectMore")).getObjectValue(s("warp")).getEnumerationValue(s("warpStyle"))!==s("warpArc")) throw Error("Authoring verification: live smart-object warp absent");
        }
        if(item.family==="style") {
            if(!d.hasKey(s("layerEffects"))) throw Error("Authoring verification: layer effects absent");
            var effects=d.getObjectValue(s("layerEffects")); key=s(item.feature+(item.variant==="multiple"?"Multi":""));
            if(!effects.hasKey(key)) throw Error("Authoring verification: requested effect "+item.feature+" absent");
            if(item.variant==="multiple"&&effects.getList(key).count<2) throw Error("Authoring verification: multiple effect instances absent");
            if(item.variant==="contour") { var storedEffect=effects.getObjectValue(key), contourKey=s(item.feature==="chromeFX"?"mappingShape":"transferSpec"); if(!storedEffect.hasKey(contourKey)||storedEffect.getObjectValue(contourKey).getString(s("name"))!=="Authored valley") throw Error("Authoring verification: requested effect contour was ignored"); }
            if(item.variant==="blend"&&effects.getObjectValue(key).getEnumerationValue(s("mode"))!==s("screen")) throw Error("Authoring verification: requested effect blend was ignored");
        }
        if(item.family==="mask") {
            if(item.feature!=="vector"&&(!d.hasKey(s("hasUserMask"))||!d.getBoolean(s("hasUserMask")))) throw Error("Authoring verification: user mask absent");
            if((item.feature==="vector"||item.feature==="user-and-vector")&&(!d.hasKey(s("hasVectorMask"))||!d.getBoolean(s("hasVectorMask")))) throw Error("Authoring verification: vector mask absent");
        }
        if(item.family==="shape"&&item.feature!=="paths-panel"&&item.feature!=="clipping-path") {
            if(!d.hasKey(s("hasVectorMask"))||!d.getBoolean(s("hasVectorMask"))) throw Error("Authoring verification: editable shape vector mask absent");
            if(item.feature.indexOf("stroke-")===0&&!d.hasKey(s("AGMStrokeStyleInfo"))) throw Error("Authoring verification: live stroke descriptor absent");
        }
        if(item.family==="metadata"&&item.feature==="count"&&doc.countItems.length!==3) throw Error("Authoring verification: count items absent");
        if(item.family==="metadata"&&item.feature==="artboard"&&!d.hasKey(s("artboard"))) throw Error("Authoring verification: artboard descriptor absent");
        if(item.family==="mode"&&item.feature==="duotone"&&doc.mode!==DocumentMode.DUOTONE) throw Error("Authoring verification: duotone mode absent");
    }
    function typeName(id) { try { var name=typeIDToStringID(id); if(name) return name; } catch(e) {} try { return typeIDToCharID(id); } catch(e2) { return String(id); } }
    function valueOf(container,key,type,depth) {
        if(depth>40) throw Error("Descriptor nesting exceeds metadata limit");
        if(type===DescValueType.BOOLEANTYPE) return container.getBoolean(key);
        if(type===DescValueType.STRINGTYPE) return container.getString(key);
        if(type===DescValueType.INTEGERTYPE) return container.getInteger(key);
        if(type===DescValueType.LARGEINTEGERTYPE) return String(container.getLargeInteger(key));
        if(type===DescValueType.DOUBLETYPE) return container.getDouble(key);
        if(type===DescValueType.UNITDOUBLE) return {unit:typeName(container.getUnitDoubleType(key)),value:container.getUnitDoubleValue(key)};
        if(type===DescValueType.ENUMERATEDTYPE) return {type:typeName(container.getEnumerationType(key)),value:typeName(container.getEnumerationValue(key))};
        if(type===DescValueType.OBJECTTYPE) return {type:typeName(container.getObjectType(key)),value:descObject(container.getObjectValue(key),depth+1)};
        if(type===DescValueType.LISTTYPE) { var list=container.getList(key), values=[]; for(var i=0;i<list.count;i++) values.push(valueOf(list,i,list.getType(i),depth+1)); return values; }
        if(type===DescValueType.CLASSTYPE) return {className:typeName(container.getClass(key))};
        if(type===DescValueType.ALIASTYPE) return {path:container.getPath(key).fsName};
        if(type===DescValueType.RAWTYPE) { var raw=container.getData(key), hex=""; for(var r=0;r<raw.length;r++) hex+=("0"+raw.charCodeAt(r).toString(16)).slice(-2); return {hex:hex}; }
        if(type===DescValueType.REFERENCETYPE) { var ref=container.getReference(key), form=ref.getForm(), obj={form:String(form)}; try { obj.className=typeName(ref.getDesiredClass()); } catch(e) {} try { obj.enumeration=typeName(ref.getEnumeratedValue()); } catch(e2) {} try { obj.property=typeName(ref.getProperty()); } catch(e3) {} try { obj.name=ref.getName(); } catch(e4) {} return obj; }
        throw Error("Unrecognized Photoshop descriptor type "+String(type));
    }
    function descObject(d,depth) { var obj={}; for(var i=0;i<d.count;i++) { var key=d.getKey(i); obj[typeName(key)]=valueOf(d,key,d.getType(key),depth||0); } return obj; }
    function property(d,key) { var id=s(key); return d.hasKey(id)?valueOf(d,id,d.getType(id),0):null; }
    function tree(doc,descriptorFolder) {
        function walk(layers,path) {
            var nodes=[];
            for(var i=0;i<layers.length;i++) {
                var l=layers[i], d=layerDescriptor(l), node={id:path.concat([i]).join("-"),ordinal:path.concat([i]),name:l.name,visible:l.visible,opacity:l.opacity,blend:String(l.blendMode),group:l.typename==="LayerSet"};
                if(descriptorFolder) write(descriptorFolder+"/"+node.id+".bin",d.toStream(),true);
                if(l.typename==="ArtLayer") { node.kind=String(l.kind); node.fill=l.fillOpacity; node.clipping=l.grouped; node.locks={all:l.allLocked,pixels:l.pixelsLocked,position:l.positionLocked,transparency:l.transparentPixelsLocked}; }
                var keys=["layerEffects","textKey","smartObject","smartObjectMore","adjustment","vectorMask","AGMStrokeStyleInfo","artboard","blendRange","channelRestrictions","knockout","color","userMaskEnabled","vectorMaskEnabled"];
                node.properties={}; for(var k=0;k<keys.length;k++) node.properties[keys[k]]=property(d,keys[k]);
                // Smart object IDs and text's derived bounds are stored in the
                // semantic descriptor intentionally; changed content is a mismatch.
                if(node.group) node.children=walk(l.layers,path.concat([i]));
                nodes.push(node);
            }
            return nodes;
        }
        var channels=[]; for(var i=0;i<doc.channels.length;i++) channels.push({name:doc.channels[i].name,kind:String(doc.channels[i].kind)});
        var guides=[]; for(var g=0;g<doc.guides.length;g++) guides.push({direction:String(doc.guides[g].direction),position:doc.guides[g].coordinate.as("px")});
        var documentTree={width:doc.width.as("px"),height:doc.height.as("px"),resolution:doc.resolution,mode:String(doc.mode),depth:String(doc.bitsPerChannel),channels:channels,guides:guides,layers:walk(doc.layers,[])};
        if(doc.countItems.length) { documentTree.counts=[]; for(var ci=0;ci<doc.countItems.length;ci++) { var cp=doc.countItems[ci].position; documentTree.counts.push([cp[0].as("px"),cp[1].as("px")]); } }
        return documentTree;
    }
    function savePng(doc,path) {
        var copy=own(doc.duplicate("pf-export",false));
        try {
            if(copy.mode===DocumentMode.BITMAP) copy.changeMode(ChangeMode.GRAYSCALE);
            if(copy.mode!==DocumentMode.RGB) copy.changeMode(ChangeMode.RGB);
            var options=new PNGSaveOptions(); options.compression=1; options.interlaced=false; copy.saveAs(new File(path),options,true,Extension.LOWERCASE);
        } finally { close(copy); app.activeDocument=doc; }
    }
    function evidence(doc,item) {
        var reference=folder((cfg.reference || cfg.output+"/reference")+"/"+item.id); folder(reference.fsName+"/layers"); folder(reference.fsName+"/descriptors");
        write(reference.fsName+"/tree.json",tree(doc,reference.fsName+"/descriptors")); savePng(doc,reference.fsName+"/composite.png");
        // Export every authored leaf in isolation while retaining its own
        // masks/styles. Adjustment layers have no standalone pixels; their
        // effect is checked by the document composite and semantic descriptor.
        var copy=own(doc.duplicate("pf-export-layers",false));
        try {
            var leaves=[], containers=[];
            function gather(layers,path,parents) { for(var i=0;i<layers.length;i++) { var layer=layers[i], at=path.concat([i]); if(layer.typename==="LayerSet") { containers.push(layer); gather(layer.layers,at,parents.concat([layer])); } else leaves.push({layer:layer,path:at,parents:parents}); } }
            gather(copy.layers,[],[]);
            for(var l=0;l<leaves.length;l++) {
                for(var h=0;h<leaves.length;h++) { try { leaves[h].layer.allLocked=false; if(h!==l&&leaves[h].layer.visible) leaves[h].layer.visible=false; } catch(e) {} }
                for(var g=0;g<containers.length;g++) if(containers[g].visible) containers[g].visible=false;
                var leaf=leaves[l]; for(var p=0;p<leaf.parents.length;p++) if(!leaf.parents[p].visible) leaf.parents[p].visible=true;
                // Flattened Bitmap/Indexed/Multichannel backgrounds reject a
                // redundant Show action. Their only leaf is already visible.
                if(!leaf.layer.visible) leaf.layer.visible=true; try { leaf.layer.grouped=false; } catch(ignore) {}
                savePng(copy,reference.fsName+"/layers/"+leaf.path.join("-")+".png");
            }
        } finally { close(copy); app.activeDocument=doc; }
    }
    function saveDocument(doc,item,destination) {
        destination=destination||cfg.output+"/originals/"+item.filename;
        if(new File(destination).exists) throw Error("Refusing to overwrite existing Photoshop output: "+destination);
        if(item.params.psb) {
            // ExtendScript has no LargeDocumentFormatSaveOptions constructor.
            // Photoshop's native ActionManager save event exposes PSB.
            var save=new ActionDescriptor(), format=new ActionDescriptor();
            format.putBoolean(s("maximizeCompatibility"),true);
            save.putObject(c("As  "),s("largeDocumentFormat"),format);
            save.putPath(c("In  "),new File(destination)); save.putBoolean(c("Cpy "),true);
            save.putBoolean(c("LwCs"),true); save.putBoolean(s("embedProfiles"),true);
            executeAction(c("save"),save,DialogModes.NO);
        }
        else { var psd=new PhotoshopSaveOptions(); psd.layers=true; psd.embedColorProfile=true; psd.alphaChannels=true; psd.spotColors=true; psd.maximizeCompatibility=true; doc.saveAs(new File(destination),psd,true,Extension.LOWERCASE); }
        return destination;
    }
    try {
        folder(cfg.output); folder(cfg.reference || cfg.output+"/reference"); folder(cfg.output+"/originals");
        if(cfg.operation==="snapshot") { snapshot={active_document_id:app.documents.length?documentID():null,documents:[]}; for(var di=0;di<app.documents.length;di++) { var documentPath=null; try { documentPath=app.documents[di].fullName.fsName; } catch(unsaved) {} snapshot.documents.push({name:app.documents[di].name,path:documentPath}); } }
        app.displayDialogs=DialogModes.NO;
        app.preferences.rulerUnits=Units.PIXELS; app.preferences.typeUnits=TypeUnits.PIXELS;
        // ERROR avoids the ordinary Open chooser forced by ALL. Its absence
        // of dialogs is not a complete warning gate: native-open verification
        // runs separately because informational warnings may be suppressed.
        app.displayDialogs=DialogModes.ERROR;
        for(var index=0;index<cfg.cases.length;index++) {
            if(cfg.abort_file&&new File(cfg.abort_file).exists) throw Error("Independent watchdog requested cancellation; remaining cases were not run");
            var item=cfg.cases[index], rec={id:item.id,family:item.family,feature:item.feature,variant:item.variant,status:"failed"};
            var started=new Date().getTime();
            try {
                progress(item,cfg.operation==="reopen"?"opening":"authoring");
                if(cfg.operation==="inspect_open") {
                    var nativeInput=new File(cfg.input+"/"+item.filename), found=null;
                    for(var nd=0;nd<app.documents.length;nd++) { try { if(app.documents[nd].fullName.fsName===nativeInput.fsName) { found=app.documents[nd]; break; } } catch(unsavedDoc) {} }
                    if(!found) throw Error("Native open did not produce the requested document: "+nativeInput.fsName);
                    current=own(found); app.activeDocument=current; rec.native_open=true;
                }
                else if(cfg.operation==="reopen") { var input=new File(cfg.input+"/"+item.filename); if(!input.exists) throw Error("Missing PixelForge roundtrip: "+input.fsName); $.hiresTimer; current=own(app.open(input)); rec.open_ms=$.hiresTimer/1000; }
                else {
                    // The DOM displays its ordinary New dialog under ALL even
                    // with complete creation arguments. Only file OPEN is the
                    // warning gate; deterministic authoring/export uses NO.
                    app.displayDialogs=DialogModes.NO;
                    current=base(item); author(current,item); verifyAuthored(current,item); progress(item,"saving");
                    var saved; $.hiresTimer; saved=saveDocument(current,item); rec.save_ms=$.hiresTimer/1000;
                    close(current); current=null; progress(item,"verifying authored file opens");
                    app.displayDialogs=DialogModes.ERROR;
                    $.hiresTimer; current=own(app.open(new File(saved))); rec.open_ms=$.hiresTimer/1000;
                }
                app.displayDialogs=DialogModes.NO;
                progress(item,"exporting evidence"); evidence(current,item);
                if(cfg.operation==="inspect_open"&&cfg.resave_output) { folder(cfg.resave_output); progress(item,"saving independent Photoshop copy"); rec.photoshop_resave_path=saveDocument(current,item,cfg.resave_output+"/"+item.filename); rec.resave_open_warning_gate_verified=false; }
                rec.status="generated";
            } catch(error) { rec.error=String(error.message||error); rec.line=error.line||null; rec.failure_kind=(/^Invalid feature combination/.test(rec.error)?"invalid_planned_feature":/^Authoring verification/.test(rec.error)?"authoring_parameter_verification":/not yet|not implemented|needs a locally|requires an authored|requires a locally|needs explicit|no built-in|authoring needs/.test(rec.error)?"unimplemented_authoring":"photoshop_api_or_io_failure"); }
            finally { if(current) { try { close(current); } catch(closeError) { rec.status="failed"; rec.error=(rec.error||"")+"; cleanup: "+closeError; } current=null; } app.displayDialogs=DialogModes.ERROR; }
            rec.seconds=(new Date().getTime()-started)/1000; results.push(rec);
            write(cfg.output+"/partial-results.json",{cases:results});
        }
    } catch(error) { fatal=String(error.message||error); }
    finally {
        var cleanup=[];
        for(var n=owned.length-1;n>=0;n--) { try { owned[n].close(SaveOptions.DONOTSAVECHANGES); } catch(e) { cleanup.push(String(e)); } }
        try { app.displayDialogs=DialogModes.NO; app.preferences.rulerUnits=preferences.ruler; app.preferences.typeUnits=preferences.type; if(cfg.previous_document_id!==undefined&&cfg.previous_document_id!==null) restoreDocument(cfg.previous_document_id); else if(previousDocument) app.activeDocument=previousDocument; app.displayDialogs=preferences.dialogs; restored=cleanup.length===0; } catch(e2) { cleanup.push(String(e2)); }
        write(cfg.output+"/photoshop-results.json",{photoshop_version:app.version,cases:results,fatal_error:fatal,restoration_confirmed:restored,cleanup_errors:cleanup,open_dialog_mode:cfg.operation==="inspect_open"?"native LaunchServices":"ERROR",warning_gate_verified:cfg.operation==="inspect_open",snapshot:snapshot});
    }
    return json({count:results.length,fatal_error:fatal,restoration_confirmed:restored});
}());

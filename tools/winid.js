// Prints the CGWindowID of the first on-screen window owned by a process
// whose name contains the argument (default "pixelforge").
ObjC.import('CoreGraphics');
ObjC.import('Foundation');
function run(argv) {
  var want = (argv[0] || 'pixelforge').toLowerCase();
  var list = ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements, 0));
  var n = list.count;
  for (var i = 0; i < n; i++) {
    var w = list.objectAtIndex(i);
    var owner = ObjC.unwrap(w.objectForKey('kCGWindowOwnerName')) || '';
    var layer = ObjC.unwrap(w.objectForKey('kCGWindowLayer'));
    if (owner.toLowerCase().indexOf(want) >= 0 && layer === 0) {
      return String(ObjC.unwrap(w.objectForKey('kCGWindowNumber')));
    }
  }
  return '';
}

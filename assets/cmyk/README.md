# CMYK readout table

`swop_v2_17.bin` is a 17 × 17 × 17 grid of the CMYK values that Photoshop 2026
reports (`SolidColor.cmyk`) for sRGB inputs, under its default colour settings:
working CMYK "U.S. Web Coated (SWOP) v2", relative colorimetric with black point
compensation.

- Grid nodes are 0, 16, 32, …, 240, 255 per channel, in R-major order (R outer,
  B inner).
- Each node holds C, M, Y and K as little-endian `u16`, where 65535 = 100 %.

It was measured with `tools/photoshop_gaps.sh <dir> F` (`info_F.txt`). The Info
panel interpolates it trilinearly, so readouts follow Photoshop's SWOP conversion
without the ICC profile. `--compare-gaps <dir> F` checks it against 200 random
colours that are not on the grid.

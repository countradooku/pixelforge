# Repeated baseline comparisons

Nine alternating repetitions per binary, suite and tier. All raw samples are retained. GPU compositor workload uses the original blend-mode sequence. Follow-up samples are separate, not replacements.

| Suite | Tier (MP) | Operation | Before median (ms) | After median (ms) | Change |
|---|---:|---|---:|---:|---:|
| engine | 24 | blank white canvas (shared tiles) | 0.388 | 0.447 | +15.21% |
| engine | 24 | procedural fill (generate every pixel) | 8.792 | 9.034 | +2.75% |
| engine | 24 | mip pyramid build | 2.730 | 2.776 | +1.68% |
| engine | 24 | brush stroke (400 events, 300 px soft) | 65.469 | 65.584 | +0.18% |
| engine | 24 | incremental mip update for stroke | 1.239 | 1.283 | +3.55% |
| engine | 24 | hue/saturation (full image) | 14.886 | 14.537 | -2.34% |
| engine | 24 | levels inside elliptical selection | 2.469 | 2.665 | +7.94% |
| engine | 24 | gaussian blur r=20 (full image) | 15.426 | 15.249 | -1.15% |
| engine | 24 | gaussian blur r=120 (full image) | 34.775 | 33.976 | -2.30% |
| engine | 24 | unsharp mask (full image) | 10.750 | 10.753 | +0.03% |
| engine | 24 | linear gradient (full image) | 6.201 | 6.024 | -2.85% |
| engine | 24 | flip horizontal | 1.758 | 1.804 | +2.62% |
| engine | 24 | rotate 90° clockwise | 1.584 | 1.584 | +0.00% |
| engine | 24 | resize to 50% (mip + bilinear) | 1.923 | 1.892 | -1.61% |
| engine | 24 | magic wand, contiguous (blank canvas) | 0.009 | 0.007 | -22.22% |
| engine | 24 | magic wand, global (procedural) | 1.119 | 1.108 | -0.98% |
| engine | 24 | flatten 2 layers (multiply blend) | 1.771 | 1.831 | +3.39% |
| engine | 24 | display: 135 screen tiles at mip level 0 | 1.541 | 1.530 | -0.71% |
| engine | 24 | display: 135 screen tiles at mip level 3 | 0.687 | 0.663 | -3.49% |
| engine | 24 | display: 135 screen tiles at mip level -2 | 0.909 | 0.873 | -3.96% |
| engine | 24 | display: live blur r=60 preview at level 2 | 5.114 | 5.239 | +2.44% |
| engine | 24 | native document save | 19.181 | 18.558 | -3.25% |
| engine | 24 | native document open | 6.954 | 6.924 | -0.43% |
| engine | 61 | blank white canvas (shared tiles) | 1.678 | 1.626 | -3.10% |
| engine | 61 | procedural fill (generate every pixel) | 21.215 | 21.558 | +1.62% |
| engine | 61 | mip pyramid build | 6.509 | 6.513 | +0.06% |
| engine | 61 | brush stroke (400 events, 300 px soft) | 90.554 | 88.519 | -2.25% |
| engine | 61 | incremental mip update for stroke | 1.781 | 1.728 | -2.98% |
| engine | 61 | hue/saturation (full image) | 34.130 | 33.634 | -1.45% |
| engine | 61 | levels inside elliptical selection | 4.239 | 4.262 | +0.54% |
| engine | 61 | gaussian blur r=20 (full image) | 32.123 | 31.940 | -0.57% |
| engine | 61 | gaussian blur r=120 (full image) | 76.864 | 77.551 | +0.89% |
| engine | 61 | unsharp mask (full image) | 26.044 | 25.885 | -0.61% |
| engine | 61 | linear gradient (full image) | 14.256 | 14.566 | +2.17% |
| engine | 61 | flip horizontal | 3.655 | 3.616 | -1.07% |
| engine | 61 | rotate 90° clockwise | 3.741 | 3.630 | -2.97% |
| engine | 61 | resize to 50% (mip + bilinear) | 4.433 | 4.196 | -5.35% |
| engine | 61 | magic wand, contiguous (blank canvas) | 0.011 | 0.007 | -36.36% |
| engine | 61 | magic wand, global (procedural) | 2.391 | 2.479 | +3.68% |
| engine | 61 | flatten 2 layers (multiply blend) | 2.767 | 2.645 | -4.41% |
| engine | 61 | display: 135 screen tiles at mip level 0 | 1.340 | 1.358 | +1.34% |
| engine | 61 | display: 135 screen tiles at mip level 3 | 0.827 | 0.849 | +2.66% |
| engine | 61 | display: 135 screen tiles at mip level -2 | 0.938 | 0.887 | -5.44% |
| engine | 61 | display: live blur r=60 preview at level 2 | 5.449 | 5.388 | -1.12% |
| engine | 61 | native document save | 26.510 | 25.693 | -3.08% |
| engine | 61 | native document open | 11.756 | 11.917 | +1.37% |
| engine | 150 | blank white canvas (shared tiles) | 1.259 | 1.321 | +4.92% |
| engine | 150 | procedural fill (generate every pixel) | 52.445 | 52.999 | +1.06% |
| engine | 150 | mip pyramid build | 14.858 | 14.883 | +0.17% |
| engine | 150 | brush stroke (400 events, 300 px soft) | 128.546 | 133.410 | +3.78% |
| engine | 150 | incremental mip update for stroke | 1.803 | 1.701 | -5.66% |
| engine | 150 | hue/saturation (full image) | 80.454 | 79.953 | -0.62% |
| engine | 150 | levels inside elliptical selection | 9.619 | 9.314 | -3.17% |
| engine | 150 | gaussian blur r=20 (full image) | 74.869 | 74.413 | -0.61% |
| engine | 150 | gaussian blur r=120 (full image) | 193.231 | 192.895 | -0.17% |
| engine | 150 | unsharp mask (full image) | 65.502 | 65.469 | -0.05% |
| engine | 150 | linear gradient (full image) | 36.716 | 37.562 | +2.30% |
| engine | 150 | flip horizontal | 9.852 | 9.753 | -1.00% |
| engine | 150 | rotate 90° clockwise | 8.802 | 8.920 | +1.34% |
| engine | 150 | resize to 50% (mip + bilinear) | 10.103 | 10.162 | +0.58% |
| engine | 150 | magic wand, contiguous (blank canvas) | 0.016 | 0.012 | -25.00% |
| engine | 150 | magic wand, global (procedural) | 5.884 | 6.010 | +2.14% |
| engine | 150 | flatten 2 layers (multiply blend) | 4.219 | 4.223 | +0.09% |
| engine | 150 | display: 135 screen tiles at mip level 0 | 0.958 | 0.854 | -10.86% |
| engine | 150 | display: 135 screen tiles at mip level 3 | 0.889 | 0.913 | +2.70% |
| engine | 150 | display: 135 screen tiles at mip level -2 | 0.997 | 0.965 | -3.21% |
| engine | 150 | display: live blur r=60 preview at level 2 | 5.778 | 5.878 | +1.73% |
| engine | 150 | native document save | 51.353 | 51.151 | -0.39% |
| engine | 150 | native document open | 23.491 | 23.541 | +0.21% |
| engine | 268 | blank white canvas (shared tiles) | 0.052 | 0.047 | -9.62% |
| engine | 268 | procedural fill (generate every pixel) | 99.980 | 99.265 | -0.72% |
| engine | 268 | mip pyramid build | 25.742 | 25.711 | -0.12% |
| engine | 268 | brush stroke (400 events, 300 px soft) | 188.754 | 184.782 | -2.10% |
| engine | 268 | incremental mip update for stroke | 2.315 | 2.269 | -1.99% |
| engine | 268 | hue/saturation (full image) | 139.925 | 138.591 | -0.95% |
| engine | 268 | levels inside elliptical selection | 14.826 | 14.739 | -0.59% |
| engine | 268 | gaussian blur r=20 (full image) | 137.417 | 137.872 | +0.33% |
| engine | 268 | gaussian blur r=120 (full image) | 344.250 | 346.139 | +0.55% |
| engine | 268 | unsharp mask (full image) | 129.115 | 127.179 | -1.50% |
| engine | 268 | linear gradient (full image) | 72.511 | 73.289 | +1.07% |
| engine | 268 | flip horizontal | 13.878 | 14.794 | +6.60% |
| engine | 268 | rotate 90° clockwise | 18.659 | 18.794 | +0.72% |
| engine | 268 | resize to 50% (mip + bilinear) | 18.055 | 18.318 | +1.46% |
| engine | 268 | magic wand, contiguous (blank canvas) | 0.040 | 0.036 | -10.00% |
| engine | 268 | magic wand, global (procedural) | 10.715 | 10.704 | -0.10% |
| engine | 268 | flatten 2 layers (multiply blend) | 6.070 | 6.168 | +1.61% |
| engine | 268 | display: 135 screen tiles at mip level 0 | 0.924 | 0.953 | +3.14% |
| engine | 268 | display: 135 screen tiles at mip level 3 | 1.143 | 1.095 | -4.20% |
| engine | 268 | display: 135 screen tiles at mip level -2 | 0.957 | 0.979 | +2.30% |
| engine | 268 | display: live blur r=60 preview at level 2 | 6.262 | 6.075 | -2.99% |
| engine | 268 | native document save | 77.226 | 76.952 | -0.35% |
| engine | 268 | native document open | 35.659 | 36.232 | +1.61% |
| tools | 24 | display: 135 tiles, mask + clip + adjustment, level 0 | 6.980 | 7.100 | +1.72% |
| tools | 24 | display: 135 tiles, mask + clip + adjustment, level 3 | 0.770 | 0.730 | -5.19% |
| tools | 24 | shape live resize (≈2000–3200 px rounded rect + stroke) | 1.250 | 1.220 | -2.40% |
| tools | 61 | display: 135 tiles, mask + clip + adjustment, level 0 | 6.890 | 6.640 | -3.63% |
| tools | 61 | display: 135 tiles, mask + clip + adjustment, level 3 | 0.710 | 0.700 | -1.41% |
| tools | 61 | shape live resize (≈2000–3200 px rounded rect + stroke) | 1.220 | 1.190 | -2.46% |
| tools | 150 | display: 135 tiles, mask + clip + adjustment, level 0 | 6.550 | 6.490 | -0.92% |
| tools | 150 | display: 135 tiles, mask + clip + adjustment, level 3 | 0.710 | 0.680 | -4.23% |
| tools | 150 | shape live resize (≈2000–3200 px rounded rect + stroke) | 1.160 | 1.180 | +1.72% |
| tools | 268 | display: 135 tiles, mask + clip + adjustment, level 0 | 6.540 | 6.420 | -1.83% |
| tools | 268 | display: 135 tiles, mask + clip + adjustment, level 3 | 0.700 | 0.700 | +0.00% |
| tools | 268 | shape live resize (≈2000–3200 px rounded rect + stroke) | 1.200 | 1.220 | +1.67% |
| gpu | 24 | compositing / CPU all tiles | 74.200 | 74.700 | +0.67% |
| gpu | 24 | compositing / GPU cold uploads all tiles | 166.400 | 167.300 | +0.54% |
| gpu | 24 | compositing / GPU warm all tiles | 7.000 | 6.800 | -2.86% |
| gpu | 24 | compositing / GPU one-tile edit | 4.800 | 4.400 | -8.33% |
| gpu | 61 | compositing / CPU all tiles | 71.700 | 73.200 | +2.09% |
| gpu | 61 | compositing / GPU cold uploads all tiles | 173.400 | 175.400 | +1.15% |
| gpu | 61 | compositing / GPU warm all tiles | 7.000 | 7.200 | +2.86% |
| gpu | 61 | compositing / GPU one-tile edit | 7.400 | 7.000 | -5.41% |
| gpu | 150 | compositing / CPU all tiles | 65.600 | 64.900 | -1.07% |
| gpu | 150 | compositing / GPU cold uploads all tiles | 143.100 | 142.900 | -0.14% |
| gpu | 150 | compositing / GPU warm all tiles | 6.900 | 6.400 | -7.25% |
| gpu | 150 | compositing / GPU one-tile edit | 9.800 | 10.500 | +7.14% |
| gpu | 268 | compositing / CPU all tiles | 54.000 | 55.500 | +2.78% |
| gpu | 268 | compositing / GPU cold uploads all tiles | 139.300 | 146.700 | +5.31% |
| gpu | 268 | compositing / GPU warm all tiles | 6.800 | 6.500 | -4.41% |
| gpu | 268 | compositing / GPU one-tile edit | 15.900 | 15.300 | -3.77% |

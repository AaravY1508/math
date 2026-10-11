# Maximum minimum light intensity: n = 37 to 50

Best placements found for [Erich Friedman's light problem](https://erich-friedman.github.io/packing/light/):
place n lights of unit brightness in the unit square to maximize the minimum light
intensity I, where the intensity at a point is the sum of the squared reciprocals of
its distances to the lights.

The archived page (retrieved 2026-10-02) lists records up to n = 36 only, so these
values have no published entry to compare against.

## Values

| n | I | certified interval for I | blue points | picture |
|---:|---:|:---|---:|:---|
| 37 | 387.711+ | [387.711396768, 387.711396868] | 48 | [37.gif](37.gif) |
| 38 | 402.793+ | [402.793547537, 402.793547635] | 44 | [38.gif](38.gif) |
| 39 | 416.038+ | [416.038294106, 416.038294207] | 44 | [39.gif](39.gif) |
| 40 | 431.089+ | [431.089121111, 431.089121211] | 42 | [40.gif](40.gif) |
| 41 | 443.973+ | [443.973887839, 443.973887940] | 46 | [41.gif](41.gif) |
| 42 | 457.633+ | [457.633089246, 457.633089346] | 48 | [42.gif](42.gif) |
| 43 | 471.024+ | [471.024139518, 471.024139618] | 44 | [43.gif](43.gif) |
| 44 | 485.241+ | [485.241524976, 485.241525077] | 46 | [44.gif](44.gif) |
| 45 | 499.219+ | [499.219199921, 499.219200022] | 51 | [45.gif](45.gif) |
| 46 | 513.635+ | [513.635468168, 513.635468269] | 52 | [46.gif](46.gif) |
| 47 | 528.179+ | [528.179126795, 528.179126895] | 50 | [47.gif](47.gif) |
| 48 | 543.672+ | [543.672931036, 543.672931137] | 48 | [48.gif](48.gif) |
| 49 | 557.579+ | [557.579125729, 557.579125829] | 59 | [49.gif](49.gif) |
| 50 | 572.393+ | [572.393899331, 572.393899431] | 56 | [50.gif](50.gif) |

**I** is shown in the page's style: the certified lower bound truncated to three
decimals, followed by `+`. The **certified interval** comes from an outward-rounded
branch-and-bound over the whole closed square for the exact binary64 coordinates (endpoints rounded
outward to nine decimals), so the
true minimum intensity of each placement lies inside it. These are feasible placements;
they are not proved globally optimal.

## Pictures

Each `n.gif` follows the format of the GIFs on the problem page (232 x 230 px):

- red dots: the lights
- black curves: intensity contours at 1.5·I and 3·I (the levels that match the
  page's own pictures for n = 30, 33, 35 and 36)
- blue dots: the points of the square where the intensity equals the minimum I
  (interior points, points on the sides, and corners)
- thick black square: the unit square

For n = 45 and n = 50, one local minimum of the intensity is 4% and 2.6% brighter
than I. Those two points are not marked, following the page's definition of the blue dots.

Full-precision coordinates and validation records for every n are in
`maximum-light-intensity/results/nN/` of the source repository. The pictures are
generated from those coordinates by `export_site_gifs.py`.

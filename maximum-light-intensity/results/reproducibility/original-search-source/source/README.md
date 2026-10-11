# Published specification and comparison snapshot

Source: <https://erich-friedman.github.io/packing/light/>.

`index.html` and `1.gif` through `36.gif` are an unchanged snapshot of the public
problem page and its configuration illustrations. `manifest.json` records the
retrieval time, source URLs, sizes, and SHA-256 hashes. Attribution remains with
the source page and its named contributors.

`published.json` is the machine-readable extraction of all 36 published entries.
The final entry is n=36, **373.207+**, attributed to Mishari Alhussain in September
2026. The page contains no entries for n=37–50. An absent entry is not interpreted
as a zero bound or evidence that no other solution exists.

A trailing `+` gives a displayed lower threshold rather than exact coordinates or
an exact objective; the `+` alone supplies no upper bound on the unpublished
objective. A difference from that displayed number must not be presented
as proof of a new record when the new number has the same displayed precision.

`seed36.json` is a numerical seed digitized from the red dots in `36.gif`.
Its 36 red and 46 blue connected-component centroids are normalized to the square
using corner-marker pixel coordinates. It preserves the pixel centroids, source
image hash, mapping, and uncertainty warning. This raster-derived seed does not
reproduce the published solution exactly and is not assigned the published
objective. Any optimized descendant must be evaluated and independently validated.

To reproduce extraction from the archived files, run `python3 baseline.py` from
the project directory. To fetch a fresh snapshot, use `python3 baseline.py --refresh`.
Preserve the existing snapshot when comparing results from a completed search.

`latest-check.json` and `latest-index.html` record a post-search source check. The
only page change was the ordering of the n=21 contributor name. All objective
values and the lack of n=37–50 entries were unchanged; the original source
snapshot remains intact.

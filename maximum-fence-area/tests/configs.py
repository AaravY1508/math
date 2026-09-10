"""Hand-constructed configurations with analytically known answers (shared by tests)."""
from mpmath import mp, mpf, sqrt

mp.dps = 60
S3_2 = mp.nstr(sqrt(3) / 2, 55)          # sqrt(3)/2 to 55 digits, as an exact decimal string


def seg(x1, y1, x2, y2):
    return [str(x1), str(y1), str(x2), str(y2)]


# n=3: unit equilateral triangle, A = sqrt(3)/4
TRIANGLE = [seg(0, 0, 1, 0), seg(1, 0, "0.5", S3_2), seg("0.5", S3_2, 0, 0)]
TRIANGLE_AREA = sqrt(3) / 4

# n=4: unit square from 4 fences, A = 1
SQUARE = [seg(0, 0, 1, 0), seg(1, 0, 1, 1), seg(1, 1, 0, 1), seg(0, 1, 0, 0)]

# n=5: unit square split by a vertical fence whose ends are T-junctions in the
# interiors of the bottom and top fences -> two faces of area 1/2, A = 1
SPLIT_SQUARE = SQUARE + [seg("0.5", 0, "0.5", 1)]

# n=7: two unit squares sharing a side, A = 2
DOMINO = [seg(0, 0, 1, 0), seg(1, 0, 2, 0), seg(0, 1, 1, 1), seg(1, 1, 2, 1),
          seg(0, 0, 0, 1), seg(1, 0, 1, 1), seg(2, 0, 2, 1)]


def grid(rows, cols):
    """rows x cols grid of unit squares from unit fences; A = rows*cols."""
    s = []
    for r in range(rows + 1):
        for c in range(cols):
            s.append(seg(c, r, c + 1, r))
    for r in range(rows):
        for c in range(cols + 1):
            s.append(seg(c, r, c, r + 1))
    return s


GRID_2x2 = grid(2, 2)      # n=12, A=4 (the site's "trivial" 12)
GRID_2x3 = grid(2, 3)      # n=17, A=6

# -------- configurations that MUST be rejected ---------------------------------

# Two fences crossing at their midpoints, both ends resting on a unit square's sides.
# If crossing were allowed this would be 4 faces of area 1/4.
CROSSING = SQUARE + [seg(0, "0.5", 1, "0.5"), seg("0.5", 0, "0.5", 1)]

# Regular hexagon of unit fences: a single face of area 3*sqrt(3)/2 ~ 2.598 > 1.
def hexagon(cx="0", cy="0"):
    pts = []
    for k in range(6):
        th = mp.pi * k / 3
        pts.append((mpf(cx) + mp.cos(th), mpf(cy) + mp.sin(th)))
    return [[mp.nstr(pts[k][0], 55), mp.nstr(pts[k][1], 55),
             mp.nstr(pts[(k + 1) % 6][0], 55), mp.nstr(pts[(k + 1) % 6][1], 55)] for k in range(6)]


HEXAGON = hexagon()
HEXAGON_AREA = 3 * sqrt(3) / 2

# Domino whose middle wall is shifted to x=1/2 with T-junction ends:
# faces 1/2 and 3/2 -> the 3/2 face violates the area constraint.
LOPSIDED = [seg(0, 0, 1, 0), seg(1, 0, 2, 0), seg(0, 1, 1, 1), seg(1, 1, 2, 1),
            seg(0, 0, 0, 1), seg("0.5", 0, "0.5", 1), seg(2, 0, 2, 1)]

# Hexagon with a unit square floating inside it: the hexagon's face has a hole.
# Faces: hexagon minus square (~1.598, violates) and the square (1). Total 2.598.
HEX_WITH_HOLE = HEXAGON + [seg("-0.5", "-0.5", "0.5", "-0.5"), seg("0.5", "-0.5", "0.5", "0.5"),
                           seg("0.5", "0.5", "-0.5", "0.5"), seg("-0.5", "0.5", "-0.5", "-0.5")]

# Square with one fence whose far end dangles in the air.
DANGLING = SQUARE + [seg(1, "0.5", 2, "0.5")]

# Two collinear fences overlapping by 1/2 (plus supports so ends are on fences).
OVERLAP = SQUARE + [seg(0, "0.5", 1, "0.5"), seg(0, "0.5", 1, "0.5")]

# Square with one side of length 1.001
BAD_LENGTH = [seg(0, 0, "1.001", 0), seg("1.001", 0, 1, 1), seg(1, 1, 0, 1), seg(0, 1, 0, 0)]

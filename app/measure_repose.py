#!/usr/bin/env python3
"""python3 measure_repose.py res/sand_collapse/frame_30.vtk [--dx 0.01]"""
import sys, re, math
import numpy as np

def read_vtk_points(path):
    # learnSPH writes legacy VTK, BINARY, big-endian doubles
    with open(path, "rb") as f:
        raw = f.read()
    m = re.search(rb"POINTS\s+(\d+)\s+double\s*\n", raw)
    n = int(m.group(1)); off = m.end()
    return np.frombuffer(raw[off: off + 24 * n], dtype=">f8").reshape(n, 3).astype(float)

def main():
    path = sys.argv[1]
    dx = float(sys.argv[sys.argv.index("--dx") + 1]) if "--dx" in sys.argv else 0.01
    P = read_vtk_points(path)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    rho = np.hypot(x, y)                       # radial height profile around the axis
    nb = int(rho.max() / dx) + 2
    h = np.zeros(nb)
    for ri, zi in zip(rho, z):
        b = int(ri / dx); h[b] = max(h[b], zi)
    rc = (np.arange(nb) + 0.5) * dx
    hmax = h.max()

    sel = (h > 0.2 * hmax) & (h < 0.8 * hmax) & (rc > 0)   # flank fit
    angle = math.degrees(math.atan(-np.polyfit(rc[sel], h[sel], 1)[0])) if sel.sum() >= 3 else float("nan")

    print(f"particles        : {len(P)}")
    print(f"final height     : {hmax:.4f} m")
    print(f"runout radius    : {rho.max():.4f} m")
    print(f"flank slope angle: {angle:.1f} deg")

if __name__ == "__main__":
    main()
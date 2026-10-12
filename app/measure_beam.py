#!/usr/bin/env python3
"""python3 measure_beam.py res/cantilever/frame_30.vtk [--dx 0.01]
Tip deflection + two fracture indicators (max nearest-neighbour distance, neighbour deficiency)."""
import sys, re
import numpy as np
from scipy.spatial import cKDTree

def read_vtk_points(path):
    raw = open(path, "rb").read()
    m = re.search(rb"POINTS\s+(\d+)\s+double\s*\n", raw)
    n = int(m.group(1)); off = m.end()
    return np.frombuffer(raw[off: off + 24 * n], dtype=">f8").reshape(n, 3).astype(float)

def main():
    path = sys.argv[1]
    dx = float(sys.argv[sys.argv.index("--dx") + 1]) if "--dx" in sys.argv else 0.01
    P = read_vtk_points(path)
    tree = cKDTree(P)
    nn = tree.query(P, k=2)[0][:, 1]
    support = 2.4 * dx                      # 2h, h = 2.4 r = 1.2 dx
    counts = np.array([len(c) - 1 for c in tree.query_ball_point(P, support)])

    free = P[P[:, 0] > 0]
    tip = free[free[:, 0] > free[:, 0].max() - dx]
    z0 = 0.5 + 0.05
    print(f"particles                : {len(P)}")
    print(f"tip deflection           : {z0 - tip[:, 2].mean():.4f} m")
    print(f"min / max NN distance    : {nn.min()/dx:.2f} dx / {nn.max()/dx:.2f} dx   (healthy ~0.9-1.1 dx)")
    print(f"particles with NN > 1.5dx: {(nn > 1.5*dx).sum()}")
    print(f"particles with < 15 nbrs : {(counts < 15).sum()}   (median {int(np.median(counts))})")
    print("verdict:", "FRACTURED" if (nn > 1.5*dx).sum() > 0 or (counts < 15).sum() > 20 else "intact")

if __name__ == "__main__":
    main()
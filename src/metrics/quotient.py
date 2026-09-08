"""Piecewise-affine polygon image graph, with explicit numerical tolerances.

This computes the image of a polygonal circle, not an exact smooth-torus
quotient and not the full Beshkov overlap complex. ReLU breakpoints are
resolved on every input edge; pairwise segment overlaps are solved by LP.
"""
import numpy as np
from scipy.optimize import linprog


def split_relu_segment(a, b, layers, tolerance=1e-9):
    segments = [(np.asarray(a, float), np.asarray(b, float))]
    for weight, bias, relu in layers:
        next_segments = []
        for left, right in segments:
            x, y = weight@left+bias, weight@right+bias
            if not relu:
                next_segments.append((x, y)); continue
            diff = y-x
            active = np.abs(diff) > tolerance
            roots = -x[active]/diff[active]
            cuts = np.unique(np.r_[0., roots[(roots > tolerance) & (roots < 1-tolerance)], 1.])
            for t, u in zip(cuts[:-1], cuts[1:]):
                next_segments.append((np.maximum(x+t*diff, 0), np.maximum(x+u*diff, 0)))
        segments = next_segments
    return segments


def image_graph(vertices, layers, tolerance=1e-7):
    segments = []
    for i in range(len(vertices)):
        segments.extend(split_relu_segment(vertices[i], vertices[(i+1)%len(vertices)], layers))
    cuts = [set([0., 1.]) for _ in segments]
    overlaps = 0
    for i, (a, b) in enumerate(segments):
        for j in range(i):
            c, d = segments[j]
            equality = np.stack((b-a, c-d), 1)
            # Both extrema are necessary for collinear overlap intervals.
            solutions = []
            for objective in ([1., 0.], [-1., 0.], [0., 1.], [0., -1.]):
                solution = linprog(objective, A_eq=equality, b_eq=c-a,
                                   bounds=[(0, 1), (0, 1)], method="highs")
                if solution.success and np.linalg.norm(equality@solution.x-(c-a)) <= tolerance:
                    solutions.append(solution.x)
            if solutions:
                overlaps += 1
                for t, u in solutions:
                    cuts[i].add(float(np.clip(t, 0, 1))); cuts[j].add(float(np.clip(u, 0, 1)))
    nodes, edges = [], set()
    def node(x):
        for i, y in enumerate(nodes):
            if np.linalg.norm(x-y) <= tolerance:
                return i
        nodes.append(x); return len(nodes)-1
    for (a, b), points in zip(segments, cuts):
        points = sorted(points)
        for t, u in zip(points[:-1], points[1:]):
            i, j = node(a+t*(b-a)), node(a+u*(b-a))
            if i != j:
                edges.add(tuple(sorted((i, j))))
    parent = list(range(len(nodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    b0 = len({find(i) for i in range(len(nodes))})
    return dict(beta0=b0, beta1=len(edges)-len(nodes)+b0, nodes=len(nodes), edges=len(edges),
                affine_segments=len(segments), pair_intersections=overlaps, tolerance=tolerance,
                domain="polygonal circle", method="ReLU edge subdivision + LP intersections + image graph")

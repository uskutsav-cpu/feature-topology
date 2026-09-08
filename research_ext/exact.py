"""Exact rational analysis of SMALL piecewise-affine ReLU networks.

The domain is a finite union of straight polygon edges, NOT a smooth torus.
Float weights, when explicitly exported, denote their exact stored rational
values evaluated with ideal arithmetic, NOT floating-point execution semantics.
This Python checker is not a Lean proof and does not claim to be one.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import combinations
import math
from typing import Iterable, Sequence

Point = tuple[Q, ...]
Matrix = tuple[Point, ...]


def rational(value) -> Q:
    if isinstance(value, bool):
        raise ValueError("Booleans are not rational coefficients")
    if isinstance(value, (int, Q)):
        return Q(value)
    if isinstance(value, str):
        if len(value) > 10000:
            raise ValueError("Rational literal too large")
        return Q(value)
    raise ValueError("Use integers or rational strings, not JSON floats")


def stored_float(value: float) -> Q:
    if not math.isfinite(value):
        raise ValueError("Cannot certify nonfinite weights")
    return Q(*float(value).as_integer_ratio())


def point(values: Iterable) -> Point:
    values = tuple(rational(v) for v in values)
    if not values:
        raise ValueError("Empty vector")
    return values


def encode_point(p: Point) -> list[str]:
    return [str(v) for v in p]


def add(a: Point, b: Point) -> Point:
    if len(a) != len(b):
        raise ValueError("Dimension mismatch")
    return tuple(x+y for x, y in zip(a, b))


def sub(a: Point, b: Point) -> Point:
    return add(a, tuple(-x for x in b))


def lerp(a: Point, b: Point, t: Q) -> Point:
    return add(a, tuple(t*x for x in sub(b, a)))


def matvec(a: Matrix, x: Point) -> Point:
    if not a or any(len(row) != len(x) for row in a):
        raise ValueError("Matrix/vector dimension mismatch")
    return tuple(sum((u*v for u, v in zip(row, x)), Q(0)) for row in a)


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b or any(len(r) != len(b) for r in a) or any(len(r) != len(b[0]) for r in b):
        raise ValueError("Matrix product dimension mismatch")
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(len(b))), Q(0))
                       for j in range(len(b[0]))) for i in range(len(a)))


def identity(n: int) -> Matrix:
    return tuple(tuple(Q(i == j) for j in range(n)) for i in range(n))


@dataclass(frozen=True)
class Layer:
    weight: Matrix
    bias: Point
    relu: bool = True

    def __post_init__(self):
        if (not self.weight or len(self.weight) != len(self.bias) or not self.weight[0]
                or any(len(row) != len(self.weight[0]) for row in self.weight)):
            raise ValueError("Invalid affine layer dimensions")
        if not isinstance(self.relu, bool):
            raise ValueError("relu must be boolean")
        if any(not isinstance(v, Q) for row in self.weight for v in row) or any(not isinstance(v, Q) for v in self.bias):
            raise ValueError("Layer coefficients must be Fraction values")

    @classmethod
    def from_dict(cls, data: dict) -> "Layer":
        if len(data["weight"]) > 1024 or sum(map(len, data["weight"])) > 100000:
            raise ValueError("Exact checker is restricted to small networks")
        return cls(tuple(point(row) for row in data["weight"]), point(data["bias"]), data.get("relu", True))

    def to_dict(self) -> dict:
        return {"weight": [encode_point(row) for row in self.weight],
                "bias": encode_point(self.bias), "relu": self.relu}


def validate_network(layers: Sequence[Layer], dimension: int) -> None:
    if len(layers) > 32:
        raise ValueError("Exact checker depth limit is 32")
    for layer in layers:
        if len(layer.weight[0]) != dimension:
            raise ValueError("Consecutive layer dimension mismatch")
        dimension = len(layer.bias)


def forward(x: Point, layers: Sequence[Layer]) -> Point:
    validate_network(layers, len(x))
    for layer in layers:
        x = add(matvec(layer.weight, x), layer.bias)
        if layer.relu:
            x = tuple(max(v, Q(0)) for v in x)
    return x


@dataclass(frozen=True)
class Piece:
    edge: int
    t0: Q
    t1: Q
    left: Point
    right: Point


def split_polygon(vertices: Sequence[Point], layers: Sequence[Layer], *, max_segments: int = 512) -> list[Piece]:
    if len(vertices) < 2 or any(len(v) != len(vertices[0]) for v in vertices):
        raise ValueError("A polygon needs at least two equally sized vertices")
    validate_network(layers, len(vertices[0]))
    pieces = [Piece(i, Q(0), Q(1), v, vertices[(i+1) % len(vertices)]) for i, v in enumerate(vertices)]
    if len(pieces) > max_segments:
        raise ValueError("Exact segment budget exceeded")
    for layer in layers:
        refined = []
        for p in pieces:
            a, b = add(matvec(layer.weight, p.left), layer.bias), add(matvec(layer.weight, p.right), layer.bias)
            cuts = {Q(0), Q(1)}
            if layer.relu:
                for x, y in zip(a, b):
                    if y != x:
                        root = -x/(y-x)
                        if 0 < root < 1:
                            cuts.add(root)
            ordered = sorted(cuts)
            for t, u in zip(ordered, ordered[1:]):
                left, right = lerp(a, b, t), lerp(a, b, u)
                if layer.relu:
                    left = tuple(max(v, Q(0)) for v in left)
                    right = tuple(max(v, Q(0)) for v in right)
                refined.append(Piece(p.edge, p.t0+t*(p.t1-p.t0), p.t0+u*(p.t1-p.t0), left, right))
                if len(refined) > max_segments:
                    raise ValueError("Exact segment budget exceeded; no partial certificate is returned")
        pieces = refined
    return pieces


def point_parameter(p: Point, a: Point, b: Point) -> Q | None:
    r = sub(b, a)
    axis = next((i for i, v in enumerate(r) if v), None)
    if axis is None:
        return Q(0) if p == a else None
    t = (p[axis]-a[axis])/r[axis]
    return t if 0 <= t <= 1 and lerp(a, b, t) == p else None


def intersections(a: Point, b: Point, c: Point, d: Point) -> list[tuple[Q, Q]]:
    """Exact segment intersection endpoints in ANY finite output dimension.

    For a collinear overlap return both endpoint parameter pairs. Degenerate
    segments return a representative; input-fiber checks handle collapsed pieces.
    """
    if len({len(a), len(b), len(c), len(d)}) != 1:
        raise ValueError("Dimension mismatch")
    if any(max(min(x, y), min(z, w)) > min(max(x, y), max(z, w))
           for x, y, z, w in zip(a, b, c, d)):
        return []
    r, s, q = sub(b, a), sub(d, c), sub(c, a)
    if not any(r):
        u = point_parameter(a, c, d)
        return [(Q(0), u)] if u is not None else []
    if not any(s):
        t = point_parameter(c, a, b)
        return [(t, Q(0))] if t is not None else []
    for i, j in combinations(range(len(a)), 2):
        determinant = r[j]*s[i]-r[i]*s[j]
        if determinant:
            t = (-q[i]*s[j]+q[j]*s[i])/determinant
            u = (r[i]*q[j]-r[j]*q[i])/determinant
            if 0 <= t <= 1 and 0 <= u <= 1 and lerp(a, b, t) == lerp(c, d, u):
                return [(t, u)]
            return []
    axis = next(i for i, v in enumerate(r) if v)
    alpha, beta = q[axis]/r[axis], s[axis]/r[axis]
    if lerp(a, b, alpha) != c or lerp(a, b, alpha+beta) != d:
        return []
    low, high = max(Q(0), min(alpha, alpha+beta)), min(Q(1), max(alpha, alpha+beta))
    if low > high:
        return []
    return [(t, (t-alpha)/beta) for t in sorted({low, high})]


def graph_from_segments(segments: Sequence[tuple[Point, Point]], *, max_pairs: int = 150000) -> dict:
    if not segments:
        raise ValueError("Cannot certify an empty image")
    if len(segments)*(len(segments)-1)//2 > max_pairs:
        raise ValueError("Pairwise intersection budget exceeded")
    cuts = [{Q(0), Q(1)} for _ in segments]
    for i, (a, b) in enumerate(segments):
        for j in range(i):
            for t, u in intersections(a, b, *segments[j]):
                cuts[i].add(t)
                cuts[j].add(u)
    nodes, raw_edges = set(), set()
    for (a, b), ts in zip(segments, cuts):
        pts = [lerp(a, b, t) for t in sorted(ts)]
        nodes.update(pts)
        for p, q in zip(pts, pts[1:]):
            if p != q:
                raw_edges.add(tuple(sorted([p, q])))
    ordered = sorted(nodes)
    ids = {p:i for i, p in enumerate(ordered)}
    edges = sorted((ids[a], ids[b]) for a, b in raw_edges)
    labels = list(range(len(nodes)))
    def find(i):
        while labels[i] != i:
            labels[i] = labels[labels[i]]
            i = labels[i]
        return i
    for i, j in edges:
        labels[find(i)] = find(j)
    components = len({find(i) for i in range(len(nodes))})
    return {"vertices": [encode_point(v) for v in ordered], "edges": [list(e) for e in edges],
            "vertex_count": len(nodes), "edge_count": len(edges),
            "beta0": components, "beta1": len(edges)-len(nodes)+components}


def collision_witness(vertices: Sequence[Point], pieces: Sequence[Piece], layers: Sequence[Layer]) -> dict | None:
    def source(piece, t):
        u = piece.t0+t*(piece.t1-piece.t0)
        return lerp(vertices[piece.edge], vertices[(piece.edge+1) % len(vertices)], u)
    def witness(x, y):
        if x == y:
            return None
        fx, fy = forward(x, layers), forward(y, layers)
        if fx != fy:
            raise ArithmeticError("Invalid collision candidate")
        return {"x": encode_point(x), "y": encode_point(y), "common_output": encode_point(fx)}
    for i, p in enumerate(pieces):
        if p.left == p.right:
            found = witness(source(p, Q(0)), source(p, Q(1)))
            if found:
                return found
        for q in pieces[:i]:
            for t, u in intersections(p.left, p.right, q.left, q.right):
                found = witness(source(p, t), source(q, u))
                if found:
                    return found
    return None


def polygon_certificate(vertices: Sequence[Point], layers: Sequence[Layer], *, max_segments: int = 512) -> dict:
    pieces = split_polygon(vertices, layers, max_segments=max_segments)
    graph = graph_from_segments([(p.left, p.right) for p in pieces])
    collision = collision_witness(vertices, pieces, layers)
    return {"schema": "feature-topology.exact-polygon.v1",
            "vertices": [encode_point(v) for v in vertices],
            "layers": [layer.to_dict() for layer in layers],
            "affine_pieces": len(pieces), "graph": graph, "collision": collision,
            "injective_on_polygon_subset": collision is None,
            "evidence_level": "exact_rational_python_replay",
            "scope": "Union of the listed straight input edges under ideal rational arithmetic; not the smooth torus.",
            "lean_verified": False}


def verify_polygon_certificate(cert: dict, *, max_segments: int = 512) -> bool:
    if cert.get("schema") != "feature-topology.exact-polygon.v1":
        raise ValueError("Unsupported certificate schema")
    result = polygon_certificate([point(v) for v in cert["vertices"]],
                                 [Layer.from_dict(v) for v in cert["layers"]], max_segments=max_segments)
    # Provenance may be added by the exporter but never changes checked claims.
    return all(cert.get(k) == v for k, v in result.items())


def inverse(a: Matrix) -> Matrix | None:
    n = len(a)
    if not n or any(len(row) != n for row in a):
        raise ValueError("Inverse requires a square matrix")
    rows = [list(row)+list(unit) for row, unit in zip(a, identity(n))]
    for col in range(n):
        pivot = next((i for i in range(col, n) if rows[i][col]), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [x/divisor for x in rows[col]]
        for i in range(n):
            if i != col:
                multiplier = rows[i][col]
                rows[i] = [x-multiplier*y for x, y in zip(rows[i], rows[col])]
    return tuple(tuple(row[n:]) for row in rows)


def left_inverse(a: Matrix) -> Matrix | None:
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("Invalid matrix")
    transpose = tuple(tuple(row[j] for row in a) for j in range(len(a[0])))
    gram_inverse = inverse(matmul(transpose, a))
    if gram_inverse is None:
        return None
    result = matmul(gram_inverse, transpose)
    if matmul(result, a) != identity(len(a[0])):
        raise ArithmeticError("Left-inverse verification failed")
    return result


def certify_box(layers: Sequence[Layer], lower: Point, upper: Point) -> dict:
    if len(lower) != len(upper) or any(a > b for a, b in zip(lower, upper)):
        raise ValueError("Invalid box")
    validate_network(layers, len(lower))
    a, b = identity(len(lower)), tuple(Q(0) for _ in lower)
    signs = []
    for index, layer in enumerate(layers):
        a, b = matmul(layer.weight, a), add(matvec(layer.weight, b), layer.bias)
        active = []
        for neuron, (row, bias) in enumerate(zip(a, b)):
            lo = bias + sum((min(w*l, w*u) for w, l, u in zip(row, lower, upper)), Q(0))
            hi = bias + sum((max(w*l, w*u) for w, l, u in zip(row, lower, upper)), Q(0))
            if not layer.relu or lo >= 0:
                active.append(True)
            elif hi <= 0:
                active.append(False)
            else:
                return {"status": "inconclusive_activation_boundary", "layer": index,
                        "neuron": neuron, "lower_bound": str(lo), "upper_bound": str(hi),
                        "lean_verified": False}
        signs.append(active)
        a = tuple(row if keep else tuple(Q(0) for _ in lower) for row, keep in zip(a, active))
        b = tuple(value if keep else Q(0) for value, keep in zip(b, active))
    inverse_matrix = left_inverse(a)
    return {"schema": "feature-topology.exact-box.v1",
            "status": "certified_injective_on_box" if inverse_matrix is not None else "inconclusive_rank_deficient",
            "lower": encode_point(lower), "upper": encode_point(upper),
            "layers": [v.to_dict() for v in layers], "activation_signs": signs,
            "affine_matrix": [encode_point(row) for row in a], "affine_bias": encode_point(b),
            "left_inverse": [encode_point(row) for row in inverse_matrix] if inverse_matrix else None,
            "evidence_level": "exact_rational_python_replay", "lean_verified": False,
            "scope": "Fixed activation region on the supplied ambient-coordinate box; not global torus injectivity."}


def verify_box_certificate(cert: dict) -> bool:
    if cert.get("schema") != "feature-topology.exact-box.v1":
        raise ValueError("Unsupported certificate schema")
    result = certify_box([Layer.from_dict(v) for v in cert["layers"]], point(cert["lower"]), point(cert["upper"]))
    return all(cert.get(k) == v for k, v in result.items())

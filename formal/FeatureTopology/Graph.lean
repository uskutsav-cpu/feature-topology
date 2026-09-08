import Std

/-!
A finite, executable undirected-graph checker. It independently recomputes graph
connectivity and the combinatorial expression E + C - V. This is deliberately
NOT a proof of equivalence between this expression and singular homology, nor a
proof that Python's network-to-graph construction is correct.
-/
namespace FeatureTopology

structure FiniteGraph where
  vertices : Nat
  edges : List (Nat × Nat)
  deriving Repr, DecidableEq

def validGraph (g : FiniteGraph) : Bool :=
  g.edges.all (fun e => decide (e.1 < e.2 ∧ e.2 < g.vertices)) &&
  decide g.edges.Nodup

def adjacent (g : FiniteGraph) (i j : Nat) : Bool :=
  (i == j) || g.edges.any (fun e =>
    ((e.1 == i) && (e.2 == j)) || ((e.1 == j) && (e.2 == i)))

def lookup (table : List (List Bool)) (i j : Nat) : Bool :=
  (table.getD i []).getD j false

def reachability (g : FiniteGraph) : List (List Bool) :=
  let ids := List.range g.vertices
  let initial := ids.map (fun i => ids.map (fun j => adjacent g i j))
  ids.foldl (fun previous k => ids.map (fun i => ids.map (fun j =>
    lookup previous i j || (lookup previous i k && lookup previous k j)))) initial

def components (g : FiniteGraph) : Nat :=
  let connected := reachability g
  ((List.range g.vertices).filter (fun i =>
    (List.range i).all (fun j => !(lookup connected i j)))).length

def graphCounts (g : FiniteGraph) : Nat × Nat :=
  let c := components g
  (c, g.edges.length + c - g.vertices)

def checkGraph (g : FiniteGraph) (expectedComponents expectedCycleRank : Nat) : Bool :=
  validGraph g &&
  decide (g.vertices ≤ g.edges.length + components g) &&
  decide (graphCounts g = (expectedComponents, expectedCycleRank))

end FeatureTopology

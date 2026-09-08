import FeatureTopology.Basic
import FeatureTopology.Graph

namespace FeatureTopology

def reluInt (x : Int) : Int := if x ≤ 0 then 0 else x

def reluPair (x : Int × Int) : Int × Int := (reluInt x.1, reluInt x.2)

theorem relu_has_explicit_collision : HasCollision reluPair := by
  refine ⟨(-1, 1), (0, 1), ?_, ?_⟩ <;> decide

theorem relu_is_not_injective : ¬ Injective reluPair :=
  collision_not_injective reluPair relu_has_explicit_collision

def taskProjection (x : Nat × Nat) : Nat := x.1

def nuisanceFactor (x : Nat × Nat) : Nat := x.2

theorem projection_loses_nuisance : ¬ Recoverable taskProjection nuisanceFactor := by
  apply different_factor_collision_blocks_decoder taskProjection nuisanceFactor (0, 0) (0, 1)
  · rfl
  · decide

def squareGraph : FiniteGraph :=
  ⟨4, [(0, 1), (0, 3), (1, 2), (2, 3)]⟩

def intervalGraph : FiniteGraph := ⟨2, [(0, 1)]⟩

def pointGraph : FiniteGraph := ⟨1, []⟩

theorem square_graph_counts : checkGraph squareGraph 1 1 = true := by decide

theorem interval_graph_counts : checkGraph intervalGraph 1 0 = true := by decide

theorem point_graph_counts : checkGraph pointGraph 1 0 = true := by decide

end FeatureTopology

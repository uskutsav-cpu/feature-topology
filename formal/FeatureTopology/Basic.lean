import Std

/-!
Logical foundations only. These statements do not assert that a trained network
is injective. Assumptions such as a reconstruction decoder must be established
separately for the network and domain in question.
-/
namespace FeatureTopology

universe u v w

def Injective {X : Type u} {Y : Type v} (f : X → Y) : Prop :=
  ∀ x y, f x = f y → x = y

def HasCollision {X : Type u} {Y : Type v} (f : X → Y) : Prop :=
  ∃ x y, x ≠ y ∧ f x = f y

def Recoverable {X : Type u} {Y : Type v} {Z : Type w}
    (f : X → Y) (factor : X → Z) : Prop :=
  ∃ decoder : Y → Z, ∀ x, decoder (f x) = factor x

theorem injective_of_reconstruction {X : Type u} {Y : Type v}
    (f : X → Y) (decoder : Y → X)
    (reconstruct : ∀ x, decoder (f x) = x) : Injective f := by
  intro x y equal
  calc
    x = decoder (f x) := (reconstruct x).symm
    _ = decoder (f y) := congrArg decoder equal
    _ = y := reconstruct y

theorem collision_not_injective {X : Type u} {Y : Type v}
    (f : X → Y) (collision : HasCollision f) : ¬ Injective f := by
  intro injective
  obtain ⟨x, y, different, equal⟩ := collision
  exact different (injective x y equal)

theorem injective_no_collision {X : Type u} {Y : Type v}
    (f : X → Y) (injective : Injective f) : ¬ HasCollision f := by
  intro collision
  exact collision_not_injective f collision injective

theorem collision_survives_postprocessing {X : Type u} {Y : Type v} {Z : Type w}
    (f : X → Y) (g : Y → Z) (collision : HasCollision f) :
    HasCollision (fun x => g (f x)) := by
  obtain ⟨x, y, different, equal⟩ := collision
  exact ⟨x, y, different, congrArg g equal⟩

theorem different_factor_collision_blocks_decoder
    {X : Type u} {Y : Type v} {Z : Type w}
    (f : X → Y) (factor : X → Z) (x y : X)
    (sameRepresentation : f x = f y) (differentFactor : factor x ≠ factor y) :
    ¬ Recoverable f factor := by
  intro recovery
  obtain ⟨decoder, specification⟩ := recovery
  apply differentFactor
  calc
    factor x = decoder (f x) := (specification x).symm
    _ = decoder (f y) := congrArg decoder sameRepresentation
    _ = factor y := specification y

theorem recoverable_before_postprocessing
    {X : Type u} {Y : Type v} {Z : Type w} {W : Type}
    (f : X → Y) (g : Y → Z) (factor : X → W)
    (recovery : Recoverable (fun x => g (f x)) factor) : Recoverable f factor := by
  obtain ⟨decoder, specification⟩ := recovery
  exact ⟨fun y => decoder (g y), specification⟩

theorem injective_composition {X : Type u} {Y : Type v} {Z : Type w}
    (f : X → Y) (g : Y → Z) (hf : Injective f) (hg : Injective g) :
    Injective (fun x => g (f x)) := by
  intro x y equal
  exact hf x y (hg (f x) (f y) equal)

end FeatureTopology

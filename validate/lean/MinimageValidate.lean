import Mathlib.Tactic
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.Notation

/-!
The orthorhombic wrap `d - L ⌊d/L + 1/2⌋` and one Delone step.

Smith's half-altitude test is a separate sufficient condition. These
lemmas are the wrap and the Selling step the closest-point path uses
after Lagrange size reduction.
-/

/-- Residual of `d - L ⌊d/L + 1/2⌋` lies in `[-L/2, L/2)`. -/
theorem wrap_mem {L d : ℝ} (hL : 0 < L) :
    -L / 2 ≤ d - L * (⌊d / L + (1 : ℝ) / 2⌋ : ℝ) ∧
      d - L * (⌊d / L + (1 : ℝ) / 2⌋ : ℝ) < L / 2 := by
  set x : ℝ := d / L
  set n : ℤ := ⌊x + 1 / 2⌋
  have hle : (n : ℝ) ≤ x + 1 / 2 := Int.floor_le (x + 1 / 2)
  have hlt : x + 1 / 2 < (n : ℝ) + 1 := Int.lt_floor_add_one (x + 1 / 2)
  have hw : d - L * (n : ℝ) = L * (x - (n : ℝ)) := by
    unfold x
    field_simp
  have hlo : -(1 : ℝ) / 2 ≤ x - (n : ℝ) := by linarith
  have hhi : x - (n : ℝ) < (1 : ℝ) / 2 := by linarith
  have hscale_lo : -L / 2 = L * (-(1 : ℝ) / 2) := by ring
  have hscale_hi : L / 2 = L * ((1 : ℝ) / 2) := by ring
  constructor
  · rw [hw, hscale_lo]
    exact mul_le_mul_of_nonneg_left hlo hL.le
  · rw [hw, hscale_hi]
    exact mul_lt_mul_of_pos_left hhi hL

/-- `d = L (k + 1/2)` maps to `-L/2`, not `+L/2`. -/
theorem wrap_at_half {L : ℝ} (hL : 0 < L) (k : ℤ) :
    L * ((k : ℝ) + 1 / 2) - L * (⌊L * ((k : ℝ) + 1 / 2) / L + (1 : ℝ) / 2⌋ : ℝ)
      = -L / 2 := by
  have hd : L * ((k : ℝ) + 1 / 2) / L = (k : ℝ) + 1 / 2 := by
    field_simp
  have hshift : L * ((k : ℝ) + 1 / 2) / L + (1 : ℝ) / 2 = (k : ℝ) + 1 := by
    rw [hd]
    ring
  have hfloor : ⌊L * ((k : ℝ) + 1 / 2) / L + (1 : ℝ) / 2⌋ = k + 1 := by
    rw [hshift]
    exact_mod_cast Int.floor_intCast (k + 1)
  rw [hfloor]
  push_cast
  ring

/-- One Delone step keeps the zero-sum of the superbasis. -/
theorem selling_preserves_sum {G : Type*} [AddCommGroup G] (v0 v1 v2 v3 : G) :
    (-v0) + v1 + (v2 + v0) + (v3 + v0) = v0 + v1 + v2 + v3 := by
  abel

/-- Basis change for the step on the first pair. Columns are the new
vectors in the old basis: `-e₁`, `e₂`, `e₃ + e₁`. -/
def sellingBasis : Matrix (Fin 3) (Fin 3) ℚ :=
  !![(-1 : ℚ), 0, 1; 0, 1, 0; 0, 0, 1]

theorem sellingBasis_det : sellingBasis.det = -1 := by
  simp [sellingBasis, Matrix.det_fin_three]

def dot3 (u v : Fin 3 → ℚ) : ℚ :=
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def nsq (u : Fin 3 → ℚ) : ℚ :=
  dot3 u u

def add3 (u v : Fin 3 → ℚ) : Fin 3 → ℚ :=
  fun i => u i + v i

def neg3 (u : Fin 3 → ℚ) : Fin 3 → ℚ :=
  fun i => -u i

/-- The sum of the four squared lengths drops by twice the positive pair
dot. That is the decrease Andrews, Bernstein, and Sauter record for the
negative sum of the six Selling scalars. -/
theorem selling_drops (v0 v1 v2 v3 : Fin 3 → ℚ)
    (h : v3 = fun i => -(v0 i + v1 i + v2 i)) :
    (nsq v0 + nsq v1 + nsq v2 + nsq v3)
        - (nsq (neg3 v0) + nsq v1 + nsq (add3 v2 v0) + nsq (add3 v3 v0))
      = 2 * dot3 v0 v1 := by
  simp only [nsq, dot3, add3, neg3, h]
  ring

def shellIdx : List ℤ := [(-4 : ℤ), -3, -2, -1, 0, 1, 2, 3, 4]

def orthoD2 (i j k : ℤ) : ℚ :=
  let dx : ℚ := 25 - 10 * i
  let dy : ℚ := -18 - 11 * j
  let dz : ℚ := 3 - 12 * k
  dx * dx + dy * dy + dz * dz

def orthoOk : Bool :=
  (shellIdx.all fun i =>
    shellIdx.all fun j =>
      shellIdx.all fun k => 50 ≤ orthoD2 i j k) &&
  (shellIdx.any fun i =>
    shellIdx.any fun j =>
      shellIdx.any fun k => orthoD2 i j k == 50)

theorem ortho_shell : orthoOk = true := by
  native_decide

def skewD2 (i j k : ℤ) : ℚ :=
  let px : ℚ := i + j * (99 / 100)
  let py : ℚ := j * (1 / 100)
  let pz : ℚ := k
  let dx : ℚ := (1 / 50) - px
  let dy : ℚ := (-1 / 50) - py
  let dz : ℚ := 0 - pz
  dx * dx + dy * dy + dz * dz

/-- `2(a - b)` for `b = (99/100, 1/100, 0)` is a lattice vector. -/
def skewOk : Bool :=
  shellIdx.any fun i =>
    shellIdx.any fun j =>
      shellIdx.any fun k => skewD2 i j k == 0

theorem skew_shell : skewOk = true := by
  native_decide

def dyadicD2 (i j k : ℤ) : ℚ :=
  let px : ℚ := i + j * (1 - 1 / 128)
  let py : ℚ := j * (1 / 128)
  let pz : ℚ := k
  let dx : ℚ := (1 / 64) - px
  let dy : ℚ := (-1 / 64) - py
  let dz : ℚ := 0 - pz
  dx * dx + dy * dy + dz * dz

def dyadicOk : Bool :=
  shellIdx.any fun i =>
    shellIdx.any fun j =>
      shellIdx.any fun k => dyadicD2 i j k == 0

theorem dyadic_shell : dyadicOk = true := by
  native_decide

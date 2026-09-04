"""Centralized hyperparameters and settings."""
import os

SEED = 42

DELTA = 0.10  # target 1 - delta = 90% coverage
DELTA_UNICYCLE = 0.05  # paper states delta := 0.05 for Example 1 (verified against PDF text)

CALIBRATION_SIZES = [100, 500, 1000]
N_REPEATS = 500
N_TEST = 500

LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 200
HIDDEN_DIM = 64

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data_files")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

for d in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Unicycle simulation (Example 1)
UNICYCLE_NOISE_STD = 0.01
UNICYCLE_HORIZON = 10
UNICYCLE_DT = 1.0  # NOT stated in the paper -- unspecified, treat as tunable

# Paper's stated input distribution D_phi_in for u := (px0, py0, theta0, v0):
#   px0, py0 ~ Uniform([0, 1))
#   theta0   ~ N(0, 0.1^2)
#   v0       ~ TruncatedNormal(1, 0.1^2) on [0, 2]
# Note: omega (angular velocity) is NOT part of the paper's input u -- it's a
# fixed constant in the dynamics, not sampled per example. The paper doesn't
# state its value; treat it as tunable like UNICYCLE_DT.
UNICYCLE_POS_LOW, UNICYCLE_POS_HIGH = 0.0, 1.0
UNICYCLE_THETA_MEAN, UNICYCLE_THETA_STD = 0.0, 0.1
UNICYCLE_V_MEAN, UNICYCLE_V_STD = 1.0, 0.1
UNICYCLE_V_CLIP = (0.0, 2.0)
UNICYCLE_OMEGA = 0.0  # NOT stated in the paper -- unspecified, treat as tunable

# Paper's safe set: C_out := {p | h_out(p) <= 0},
#   h_out(p) = (px - 13.5)^2 + (py - 0.5)^2 - 3^2  (disk, center (13.5, 0.5), radius 3)
# Kept for reference / optional reachability scoring; not used by the four
# regression-interval CP methods (vanilla/CQR/locally-adaptive/weighted),
# which score via Euclidean residual norm instead (see conformal/scoring.py).
UNICYCLE_SAFE_CENTER = (13.5, 0.5)
UNICYCLE_SAFE_RADIUS = 3.0

# sin(x) toy benchmark
TOY_TRAIN_MEAN, TOY_TRAIN_STD = -1.0, 1.0
TOY_SHIFT_MEAN, TOY_SHIFT_STD = 1.0, 1.0

# Parkinson's covariate shift split
PARKINSONS_AGE_THRESHOLD = 65
PARKINSONS_TARGET_COL = "total_UPDRS"
PARKINSONS_CSV_PATH = os.path.join(DATA_DIR, "parkinsons_updrs.csv")

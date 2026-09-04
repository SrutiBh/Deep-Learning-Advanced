"""UCI Parkinson's Telemonitoring loader.

Download the dataset yourself (network access here can't reach the UCI
archive) from:
  https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
Save the CSV as config.PARKINSONS_CSV_PATH (data_files/parkinsons_updrs.csv).
Expected columns include: age, sex, test_time, motor_UPDRS, total_UPDRS,
and the 16 biomedical voice measures (Jitter, Shimmer, NHR, HNR, RPDE,
DFA, PPE, ...).

Split design (all disjoint, no reuse across roles):
  Source domain (age <= threshold) is divided into:
    train           -> trains the backbone/quantile/variance networks
    val             -> validation loss during training
    cal_noshift     -> calibration set for the heteroskedasticity test
    test_noshift    -> held-out source-domain test set (heteroskedasticity test)
    cal_shift       -> calibration set for the covariate-shift test
  Target domain (age > threshold) is used whole as:
    test_shift      -> test set for the covariate-shift test
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


FEATURE_COLS = [
    "age", "sex", "test_time",
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
    "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]


def load_parkinsons_splits(seed=config.SEED):
    df = pd.read_csv(config.PARKINSONS_CSV_PATH)
    df = df.dropna(subset=FEATURE_COLS + [config.PARKINSONS_TARGET_COL])

    source = df[df["age"] <= config.PARKINSONS_AGE_THRESHOLD].reset_index(drop=True)
    target = df[df["age"] > config.PARKINSONS_AGE_THRESHOLD].reset_index(drop=True)

    def xy(frame):
        x = frame[FEATURE_COLS].values.astype(np.float32)
        y = frame[config.PARKINSONS_TARGET_COL].values.astype(np.float32)
        return x, y

    # Disjoint 5-way split of the source domain.
    train_df, rest_df = train_test_split(source, test_size=0.6, random_state=seed)
    val_df, rest2_df = train_test_split(rest_df, test_size=0.75, random_state=seed)
    cal_noshift_df, rest3_df = train_test_split(rest2_df, test_size=2 / 3, random_state=seed)
    test_noshift_df, cal_shift_df = train_test_split(rest3_df, test_size=0.5, random_state=seed)

    x_train, y_train = xy(train_df)
    x_val, y_val = xy(val_df)
    x_cal_noshift, y_cal_noshift = xy(cal_noshift_df)
    x_test_noshift, y_test_noshift = xy(test_noshift_df)
    x_cal_shift, y_cal_shift = xy(cal_shift_df)
    x_test_shift, y_test_shift = xy(target)

    scaler = StandardScaler().fit(x_train)
    tr = lambda x: scaler.transform(x).astype(np.float32)

    return {
        "train": (tr(x_train), y_train),
        "val": (tr(x_val), y_val),
        "cal_noshift": (tr(x_cal_noshift), y_cal_noshift),
        "test_noshift": (tr(x_test_noshift), y_test_noshift),
        "cal_shift": (tr(x_cal_shift), y_cal_shift),
        "test_shift": (tr(x_test_shift), y_test_shift),
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
    }

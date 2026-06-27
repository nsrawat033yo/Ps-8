from __future__ import annotations

import numpy as np

from lunar_icenav.features.texture import local_mean, local_std, robust_normalize


def sar_feature_stack(lh: np.ndarray, lv: np.ndarray) -> dict[str, np.ndarray | str]:
    """Create defensible SAR screening features from calibrated SRI intensity rasters.

    The available SRI LH/LV rasters are real-valued intensities, not complex I/Q.
    Therefore this function does not claim true CPR or DOP. It exposes a CPR-style
    channel ratio proxy and a polarization-imbalance proxy for screening only.
    """
    lh_f = lh.astype("float32")
    lv_f = lv.astype("float32")
    valid = np.isfinite(lh_f) & np.isfinite(lv_f) & (lh_f > 0) & (lv_f > 0)
    eps = np.float32(1e-6)
    intensity = np.log1p(lh_f + lv_f)
    ratio_proxy = lh_f / np.maximum(lv_f, eps)
    inverse_ratio_proxy = lv_f / np.maximum(lh_f, eps)
    pol_imbalance_proxy = np.abs(lh_f - lv_f) / np.maximum(lh_f + lv_f, eps)
    local_mean_intensity = local_mean(intensity, size=9, valid=valid)
    local_std_intensity = local_std(intensity, size=9, valid=valid)
    score = (
        0.42 * robust_normalize(ratio_proxy, valid)
        + 0.36 * robust_normalize(intensity, valid)
        + 0.14 * robust_normalize(local_std_intensity, valid)
        + 0.08 * robust_normalize(pol_imbalance_proxy, valid)
    ).astype("float32")
    score[~valid] = np.nan
    return {
        "lh": lh_f,
        "lv": lv_f,
        "valid": valid,
        "intensity": intensity.astype("float32"),
        "cpr_style_ratio_proxy": ratio_proxy.astype("float32"),
        "lv_lh_ratio_proxy": inverse_ratio_proxy.astype("float32"),
        "polarization_imbalance_proxy": pol_imbalance_proxy.astype("float32"),
        "local_mean": local_mean_intensity.astype("float32"),
        "local_std": local_std_intensity.astype("float32"),
        "texture": local_std_intensity.astype("float32"),
        "candidate_score": score,
        "feature_note": (
            "SRI LH/LV rasters are intensity products. True CPR/DOP require the exact "
            "compact-pol convention and/or complex/Stokes products, so these are "
            "screening proxies only."
        ),
    }

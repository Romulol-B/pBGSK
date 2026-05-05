import numpy as np


def _as_feature_mask(
    features: np.ndarray,
    n_features: int,
    *,
    name: str = "features",
) -> np.ndarray:
    """
    Return a one-dimensional boolean feature mask with shape ``(n_features,)``.

    NumPy boolean indexing requires the mask length to match the selected axis.
    Validating that contract at the boundary gives clearer errors than letting
    column slicing fail deeper inside the evaluator.
    """
    mask = np.asarray(features, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional boolean mask with shape ({n_features},)."
        )
    if mask.shape[0] != n_features:
        raise ValueError(
            f"{name} length must match the number of features ({n_features}); "
            f"got {mask.shape[0]}."
        )
    return mask


def _count_selected_features(features: np.ndarray) -> int:
    """Count selected features in a boolean mask."""
    return int(np.count_nonzero(features))

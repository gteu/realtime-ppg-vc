import numpy as np

def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

def scale(x, data_mean, data_std):
    """Mean/variance scaling.

    Given mean and variances, apply mean-variance normalization to data.

    Args:
        x (array): Input data
        data_mean (array): Means for each feature dimention.
        data_std (array): Standard deviation for each feature dimention.

    Returns:
        array: Scaled data.
    """
    return (x - data_mean) / _handle_zeros_in_scale(data_std, copy=False)
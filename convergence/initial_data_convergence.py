import numpy as np


def _ue_2d(x):
    comp1 = np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 1
    comp2 = np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 2
    comp3 = np.sin(np.pi * x[1])
    norm = np.sqrt(comp1**2 + comp2**2 + comp3**2)
    return np.vstack([comp1, comp2, comp3]) / norm


def _ue_3d(x):
    comp1 = np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + np.cos(np.pi * x[2]) ** 2 + 1
    comp2 = np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 2
    comp3 = np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]) ** 2 + np.sin(np.pi * x[0])
    norm = np.sqrt(comp1**2 + comp2**2 + comp3**2)
    return np.vstack([comp1, comp2, comp3]) / norm


def ue(x):
    # Shared smooth unit-length initial magnetization for 2D and 3D tests.
    dim = x.shape[0]
    if dim == 2:
        return _ue_2d(x)
    if dim == 3:
        return _ue_3d(x)
    raise ValueError(f"Unsupported spatial dimension: {dim}")

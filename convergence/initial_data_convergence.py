"""
用于对不同维度给出满足初边值条件的真解
"""
import numpy as np

def _s_3d(x):
    """三维归一化因子"""
    term1 = (np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + np.cos(np.pi * x[2]) ** 2 + 1) ** 2
    term2 = (np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 2) ** 2
    term3 = (np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]) ** 2 + np.sin(np.pi * x[0])) ** 2
    return np.sqrt(term1 + term2 + term3)


def _ue_3d(x):
    """三维真解初值"""
    comp1 = np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + np.cos(np.pi * x[2]) ** 2 + 1
    comp2 = np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 2
    comp3 = np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]) ** 2 + np.sin(np.pi * x[0])
    # 使用 vstack 确保返回形状为 (3, N)，符合 dolfinx.interpolate 要求
    return np.vstack([comp1, comp2, comp3]) / _s_3d(x)


def _s_2d(x):
    """二维归一化因子 (去除了 z 轴依赖)"""
    term1 = ( np.sin(np.pi*x[0])*np.cos(2*np.pi*x[1])+1 )**2
    term2 = ( np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1])+2 )**2
    term3 = ( np.sin(np.pi*x[1]) )**2
    return np.sqrt(term1 + term2 + term3)


def _ue_2d(x):
    """二维真解初值"""
    comp1 = np.sin(np.pi*x[0])*np.cos(2*np.pi*x[1])+1
    comp2 = np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1])+2
    comp3 = np.sin(np.pi*x[1])
    # 使用 vstack 确保返回形状为 (2, N)
    return np.vstack([comp1, comp2, comp3]) / _s_2d(x)


def ue(x):
    """
    通用初值接口。
    根据输入坐标 x 的形状自动判断是 2D 还是 3D。

    参数:
        x: 坐标数组，形状为 (dim, N)，由 dolfinx 传入
    返回:
        u: 初值数组，形状为 (dim, N)
    """
    dim = x.shape[0]
    if dim == 2:
        return _ue_2d(x)
    elif dim == 3:
        return _ue_3d(x)
    else:
        raise ValueError(f"不支持的空间维度：{dim}")
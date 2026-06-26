import numpy as np
from dolfinx import fem


def m_ini(x, b=np.pi):
    """
    u_0(x) = ( x/|x| * sin(phi(|x|)), cos(phi(|x|)) )
    phi(r) = b * r^2  (r <= 1)
           = b        (r >= 1)
    """
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    phi = np.where(r <= 1.0, b * r ** 2, b)

    num_points = len(r)
    values = np.zeros((3, num_points))

    # 6. 组装磁化矢量
    # 前两个分量：x/|x| * sin(phi)
    for i in range(num_points):
        if r[i] == 0:
            values[0, i] = 0.0
            values[1, i] = 0.0
        else:
            values[0, i] = (x[0][i] / r[i]) * np.sin(phi[i])
            values[1, i] = (x[1][i] / r[i]) * np.sin(phi[i])

        values[2, i] = np.cos(phi[i])
    # 第三个分量：cos(phi)
    values[2, :] = np.cos(phi)

    return values

# 使用示例
# mho = fem.Function(V)
# mho.interpolate(lambda x: m_ini(x, b=np.pi))
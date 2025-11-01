import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri


# 定义微分方程组
def system(t, state):
    """修改微分方程：在system函数中替换为你的具体方程"""
    x, y, z = state
    # dxdt = x * (1 - x - 0.5 * y - 0.5 * z)
    # dydt = y * (1 - y - 0.5 * x - 0.5 * z)
    # dzdt = z * (1 - z - 0.5 * x - 0.5 * y)

    dxdt = x * (1 - x) - x * y
    dydt = y * (1 - y) - x * y
    dzdt = z * (1 - z) - x * z
    return [dxdt, dydt, dzdt]


def triangular_projection(x, y):
    """# 选择二维平面投影(将z固定)"""
    z = 0.5  # 可以选择 z 的一个固定值
    dxdt, dydt, _ = system(0, [x, y, z])
    return dxdt, dydt


def generate_triangular_mesh(num_points_per_side):
    """生成等边三角形网格"""
    # 等边三角形的顶点
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    # 生成网格点
    points = []
    for i in range(num_points_per_side):
        for j in range(num_points_per_side - i):
            k = num_points_per_side - i - j - 1
            barycentric = np.array([i, j, k]) / (num_points_per_side - 1)
            point = barycentric[0] * vertices[0] + barycentric[1] * vertices[1] + barycentric[2] * vertices[2]
            points.append(point)
    points = np.array(points)
    # 创建三角剖分
    triang = tri.Triangulation(points[:, 0], points[:, 1])
    return triang, points


def fun1():
    # 生成网格点
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)

    # 计算向量场
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i, j], V[i, j] = triangular_projection(X[i, j], Y[i, j])

    # 绘制流线图
    plt.figure(figsize=(8, 8))
    plt.streamplot(X, Y, U, V, density=1.5, color="b", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Streamplot of the 3D Differential Equations (x - y projection)")
    plt.grid(True)
    plt.show()


def fun2():
    # 生成网格
    num_points_per_side = 10
    triang, points = generate_triangular_mesh(num_points_per_side)

    # 计算向量场
    U = np.zeros(points.shape[0])
    V = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        x, y = points[i]
        U[i], V[i] = triangular_projection(x, y)

    # 绘制流线图
    plt.figure(figsize=(8, 8))
    plt.triplot(triang, "ko-", lw=0.5)  # 绘制三角形网格
    plt.tricontourf(triang, np.zeros(points.shape[0]), alpha=0.2)  # 填充颜色以突出显示三角形
    plt.quiver(points[:, 0], points[:, 1], U, V, scale=10, color="b")  # 绘制向量场
    plt.title("Streamplot on Equilateral Triangular Mesh")
    plt.axis("equal")
    plt.show()


def trible_equations():
    """绘制三元方程的三维图像"""
    # 生成初始条件
    n_trajectories = 20
    initial_conditions = np.random.uniform(0, 1, size=(n_trajectories, 3))

    # 积分参数
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # 创建三维图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 对每个初始条件求解并绘制轨迹
    for initial_state in initial_conditions:
        sol = solve_ivp(system, t_span, initial_state, t_eval=t_eval)
        x, y, z = sol.y
        ax.plot(x, y, z, alpha=0.6, linewidth=0.5)

    # 设置坐标轴和标签
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def main():
    fun2()
    print("Done!")


if __name__ == "__main__":
    main()

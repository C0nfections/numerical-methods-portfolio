import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from LUDecomp import LUDecomp
"""
Solving for AT = f.
Computational grid: 5x5 squares, 6x6 points.

Nx by Ny squares => (Nx+1) x (Ny+1) points

dim(T) and dim(f) should be the same. This is equal to the number of points
we have in our grid. (length + 1) times (width + 1).
"""


def BVP_solver(num_x, num_y, alpha_value):
    Nx = num_x
    Ny = num_y
    hx = 1/Nx
    hy = 1/Ny
    num_pts = (Nx+1) * (Ny+1)

    A = np.zeros((num_pts, num_pts))  # matrix: 36x36 (A)
    T = np.zeros(num_pts)  # vector dimension 36 (solution vector, x)
    f = np.zeros(num_pts)  # vector dimension 36 (b)

    alpha = alpha_value

    def idx(i, j):
        """
        i = 0...Nx ; j = 0...Ny
        Maps indices i,j to one index: k
        i.e. (0,0)->0, (1,0)->1, ..., (Nx,0)->Nx, (0,1)->Nx+1
        """
        return j * (Nx + 1) + i

    # (delta)^2 T + alpha = 0 -> (delta)^2 T = -alpha
    for j in range(Ny+1):
        for i in range(Nx+1):  # iterate through row first then go up column
            k = idx(i, j)
            if i == 0 or i == Nx or j == 0 or j == Ny:
                # set boundary points to be 0
                A[k, k] = 1
                f[k] = 0  # alpha is 0 on boundary
            else:  # use the 5 variable stencil in lecture to solve for f
                """
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / (hx)**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / (hy)**2
                = f[i, j]
                """
                fd_weight_x = 1 / (hx ** 2)
                fd_weight_y = 1 / (hy ** 2)

                A[k, idx(i + 1, j)] = fd_weight_x  # right (T[i+1, j])
                A[k, k] = -2 * fd_weight_x - 2 * fd_weight_y  # center (T[i, j])
                A[k, idx(i-1, j)] = fd_weight_x  # left (T[i-1, j])

                A[k, idx(i, j-1)] = fd_weight_y  # up (T[i, j+1])
                A[k, idx(i, j+1)] = fd_weight_y  # down (T[i, j-1])

                f[k] = -alpha

    error_lst = [0]
    LUDecomp(A, f, num_pts, tol=1e-12, x=T, error=error_lst)
    grid = T.reshape((num_y + 1, num_x + 1))  # coverts T into a grid
    return grid

# plots the contour maps
alpha = 2.0
nodes_list = [10, 15, 25, 35]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, nodes in zip(axes, nodes_list):
    num_x = nodes - 1
    num_y = nodes - 1
    T_grid = BVP_solver(num_x, num_y, alpha)
    x = np.linspace(0.0, 1.0, nodes)
    y = np.linspace(0.0, 1.0, nodes)
    X, Y = np.meshgrid(x, y)

    cs = ax.contour(X, Y, T_grid, levels=15)
    ax.clabel(cs, inline=True, fontsize=7)

    ax.set_title(f"{nodes}x{nodes} nodes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

plt.suptitle(f"2D diffusion BVP: contour lines (alpha={alpha})")
plt.tight_layout()
plt.show()


# timing performance of LU
start_time = time.time()
BVP_solver(14, 14, 2)  # 15 nodes in each dimension
end_time = time.time()
print("This algorithm/procedure for 15 nodes in each dimension took: {:.1f} seconds to execute.".format(end_time - start_time))


start_time = time.time()
BVP_solver(19, 19, 2)  # 20 nodes in each dimension
end_time = time.time()
print("This algorithm/procedure for 20 nodes in each dimension took: {:.1f} seconds to execute.".format(end_time - start_time))


start_time = time.time()
BVP_solver(24, 24, 2)  # 25 nodes in each dimension
end_time = time.time()
print("This algorithm/procedure for 25 nodes in each dimension took: {:.1f} seconds to execute.".format(end_time - start_time))


start_time = time.time()
BVP_solver(29, 29, 2)  # 30 nodes in each dimension
end_time = time.time()
print("This algorithm/procedure for 30 nodes in each dimension took: {:.1f} seconds to execute.".format(end_time - start_time))


start_time = time.time()
BVP_solver(34, 34, 2)  # 35 nodes in each dimension
end_time = time.time()
print("This algorithm/procedure for 35 nodes in each dimension took: {:.1f} seconds to execute.".format(end_time - start_time))


start_time = time.time()
BVP_solver(39, 39, 2)  # 40 nodes in each dimension
end_time = time.time()
print("This algorithm/procedure for 40 nodes in each dimension took: {:.1f} seconds to execute.".format(end_time - start_time))


# plots the graph which measures LU performance against the size of matrix
node_sizes = np.array([15, 20, 25, 30, 35, 40])
times = np.array([1.5, 8.7, 34.3, 104.2, 268.4, 582.0])

# fit power law: log(t) = p*log(n) + c
p, c = np.polyfit(np.log(node_sizes), np.log(times), 1)

# power-law scales (both x/y axes are logged)
plt.figure()
plt.loglog(node_sizes, times, marker="o", linestyle="-", label="Measured time")
plt.loglog(node_sizes, np.exp(c) * node_sizes**p, linestyle="--", label=f"Fit: n^{p:.2f}")

plt.xlabel("Matrix size n")
plt.ylabel("Time (seconds)")
plt.title("LU Performance in Seconds vs Matrix Size (log-log scale)")
plt.legend()
plt.show()


# 5.
alpha = 20
nodes_list = [10, 15, 20, 25, 35]
M_max = 36
N_max = 36


def analytical_solution(x, y, alpha, M_max=36, N_max=36):
    X, Y = np.meshgrid(x, y)
    exact = np.zeros_like(X, dtype=float)
    ms = np.arange(1, M_max+1, 2)  # odd
    ns = np.arange(1, N_max+1, 2)  # odd

    for m in ms:
        sin_mx = np.sin(m * np.pi * X)
        for n in ns:
            coef = (16.0 * alpha) / (np.pi**4 * m * n * (m*m + n*n))
            exact += coef * sin_mx * np.sin(n * np.pi * Y)
    return exact


hs = []
errs = []
for nodes in nodes_list:
    num_x = nodes-1
    num_y = nodes-1
    numerical_solution = BVP_solver(num_x, num_y, alpha)

    x = np.linspace(0.0, 1.0, nodes)
    y = np.linspace(0.0, 1.0, nodes)

    # analytical solution on same grid
    exact_solution = analytical_solution(x, y, alpha, M_max=M_max, N_max=N_max)

    # average per grid point
    diff = numerical_solution - exact_solution
    err_avg = np.linalg.norm(diff.ravel(), 2) / diff.size  # L2 norm

    h = 1.0 / (nodes-1)
    hs.append(h)
    errs.append(err_avg)

hs = np.array(hs)
errs = np.array(errs)

plt.figure()
plt.loglog(hs, errs, "o-", label="avg L2 error per grid point")
p, c = np.polyfit(np.log(hs), np.log(errs), 1)
plt.loglog(hs, np.exp(c) * hs**p, "--", label=f"fit: ~ h^{p:.2f}")
plt.xlabel("Grid spacing h = 1/(N-1)")
plt.ylabel("Average L2 error per grid point")
plt.title(f"Error vs grid resolution using analytical solution (alpha={alpha})")
plt.legend()
plt.grid(True, which="both")
plt.show()
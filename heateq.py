import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation


def adaptive_time_step(dx, alpha, safety_factor=0.9):
    """
    Función que se asegura de mantener la estabilidad de la solución, evita 
    que la solución diverja
    """
    max_dt = dx**2 / (4 * alpha)  
    return safety_factor * max_dt


def solve_sparse_iterative(A, rhs):
    """
    Esta función es un manejo de error para la solución del sistema de ecuaciones, garantiza
    que se resuelvan sistemas con solución (convergentes)
    """
    x, info = cg(A, rhs)
    if info != 0:
        raise ValueError("Conjugate Gradient solver did not converge.")
    return x


def solve_parallel(A, rhs_list):
    """
    Usa la máxima cantidad de nodos (en una sola máquina para resolver un sistema)
    """
    results = Parallel(n_jobs=-1)(delayed(solve_sparse_iterative)(A, rhs) for rhs in rhs_list) #Esto es un lista de comprehension que divide y recolecta los cálculos
    return np.array(results)


def crank_nicolson(u, A, B, r, nx, ny):
    # Se ignoran las fronteras (Condiciones de Dirichlet)
    u_inner = u[1:-1, 1:-1]

    # Resuelve primero las rejillas en x
    rhs_x = B @ u_inner + r * (u[:-2, 1:-1] + u[2:, 1:-1])
    u_x_new = solve_parallel(A, rhs_x.T)

    # Resuelve las rejillas en y
    rhs_y = B @ u_x_new.T + r * (u[1:-1, :-2] + u[1:-1, 2:])
    u_y_new = solve_parallel(A, rhs_y)

    # Actualiza la grilla completa
    u_new = np.zeros_like(u)  # Condición de Dirichlet (u(L, y ,t) = u(x, L, t) = 0)
    u_new[1:-1, 1:-1] = u_y_new.T

    return u_new

def realtime_solution(u, A, B, r, nx, ny, nt, interval=50):
    # Formato del Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.pcolormesh(u, cmap='hot', shading='auto', vmin=0, vmax=100)
    fig.colorbar(c, ax=ax)
    ax.set_title("Heat Equation Solver")
    fig.tight_layout()

    def update(frame):
        nonlocal u
        u = crank_nicolson(u, A, B, r, nx, ny)
        c.set_array(u.ravel())
        ax.set_title(f"Time Step: {frame}")

    anim = FuncAnimation(fig, update, frames=nt, interval=100, repeat=False)
    plt.show()


def main():
    # Paramétros
    L, T = 1.0, 0.5 # T es el tiempo
    kappa = 0.01 
    nx, ny = 200, 200  # Tamaño de la rejilla
    dx, dy = L / (nx - 1), L / (ny - 1)
    dt = adaptive_time_step(dx, kappa)
    nt = int(T / dt)  # Pasos temporales a usar

    # Condiciones Iniciales
    u = np.zeros((nx, ny))
    u[nx // 4, ny // 4] = 300  # Hot spot at the center
    u[nx // 4, 3*ny // 4] = 150
    u[3*nx // 4, ny // 4] = 150
    u[3*nx // 4, 3*ny // 4] = 300

    # Configuración de las matrices para el sistema de ecuaciones
    r = kappa * dt / (2 * dx**2)
    main_diag = (1 + 2 * r) * np.ones(nx - 2)
    off_diag = -r * np.ones(nx - 3)
    A = diags([main_diag, off_diag, off_diag], [0, 1, -1]).tocsc()
    B = diags([(1 - 2 * r) * np.ones(nx - 2), r * np.ones(nx - 3), r * np.ones(nx - 3)], [0, 1, -1]).tocsc()

    # Corre la simulación en tiempo real
    realtime_solution(u, A, B, r, nx, ny, nt)


if __name__ == "__main__":
    main()


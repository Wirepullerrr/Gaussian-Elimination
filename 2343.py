from scipy.linalg import lu, solve
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, e

# Constants
L = 120  # (in) length
q = 100 / 12  # (lb/in) intensity of uniform load
E = 3.0e7  # (lb/in^2) modulus of elasticity
S = 1000  # (lb) stress at ends
I = 625  # (in^4) central moment of inertia

# Analytical solution function
def analy_sol(x):
    c = -(q * E * I) / (S ** 2)
    a = sqrt(S / (E * I))
    b = -q / (2 * S)
    c1 = c * ((1 - e ** (-a * L)) / (e ** (-a * L) - e ** (a * L)))
    c2 = c * ((e ** (a * L) - 1) / (e ** (-a * L) - e ** (a * L)))

    w = c1 * e ** (a * x) + c2 * e ** (-a * x) + b * x * (x - L) + c
    return w

def set_matrix(n, h, x_list):
    cons_A = 2 + (S / (E * I)) * h ** 2
    main_diagonal = [cons_A] * (n - 1)
    off_diagonal = [-1] * (n - 2)

    A = np.diag(main_diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)

    cons_b = (q / (2 * E * I)) * h ** 2
    b = [cons_b * x * (L - x) for x in x_list]

    return np.array(A), np.array(b)

def gaussian_elimination(matrix, vector):
    P, L, U = lu(matrix)
    Pb = np.dot(P, vector)
    y = solve(L, Pb)
    x = solve(U, y)
    return x

def plot_error_plots(all_errors, all_x_lists, K):
    plt.figure(figsize=(10, 6))
    for k in range(1, K + 1):
        plt.semilogy(all_x_lists[k - 1], all_errors[k - 1], label=f'k={k}')
    plt.xlabel('Beam Length (in)')
    plt.ylabel('Error (in)')
    plt.title('Error Plots for k=1 to 13')
    plt.legend()
    plt.grid(True)
    plt.savefig("All_Errors_Log_Scale.png")
    plt.clf()

def iteration(k, all_errors, all_x_lists):
    n = 2 ** (k + 1)
    h = L / n
    x_list = [h * i for i in range(1, n)]

    A, b = set_matrix(n, h, x_list)

    approx_w = list(gaussian_elimination(A, b))
    true_w = [analy_sol(x) for x in x_list]

    approx_w = [0] + approx_w + [0]
    true_w = [0] + true_w + [0]
    x_list = [0] + x_list + [L]

    error_list = [abs(approx - true) for approx, true in zip(approx_w, true_w)]
    all_errors.append(error_list)
    all_x_lists.append(x_list)

def main():
    K = 13  # Change this to 13 for plotting errors from k=1 to 13
    all_errors = []
    all_x_lists = []

    for k in range(1, K + 1):
        iteration(k, all_errors, all_x_lists)

    plot_error_plots(all_errors, all_x_lists, K)

if __name__ == "__main__":
    main()

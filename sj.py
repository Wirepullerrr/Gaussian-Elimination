# Group Project 2 (4A)

from scipy.linalg import lu, solve
from scipy.sparse import spdiags
from numpy.linalg import cond
from math import sqrt, e, ceil, log

import numpy as np
import matplotlib.pyplot as plt

# constants we need
L = 120 # (in) length
q = 100/12 # (lb/in) intensity of uniform load
E = 3.0e7 # (lb/in^2) modulus of elasticity
S = 1000 # (lb) stress at ends
I = 625 # (in^4) central moment of inertia

# analytical solution
def analy_sol(x):
    global L, q, E, S, I

    c = -(q*E*I)/(S**2)
    a = sqrt(S/(E*I))
    b = -q/(2*S)
    c1 = c*((1-e**(-a*L))/(e**(-a*L)-e**(a*L)))
    c2 = c*((e**(a*L)-1)/(e**(-a*L)-e**(a*L)))

    w = c1*e**(a*x) + c2*e**(-a*x) + b*x*(x-L) + c
    return w



def set_matrix(n, h, x_list):
    global L, q, E, S, I

    A = [[0 for i in range(n-1)] for j in range(n-1)]
    b = [0 for i in range(n-1)]

    # put values in A
    cons_A = 2 + (S/(E*I))*h**2
    main_diagonal = [cons_A for i in range(n-1)]
    upper_diagonal = [-1 for i in range(n-2)]
    lower_diagonal = [-1 for i in range(n-2)]

    # create a tridiagonal matrix
    A = np.diag(main_diagonal) + np.diag(upper_diagonal, k=1) + np.diag(lower_diagonal, k=-1)

    # put values in b
    cons_b = (q/(2*E*I))*h**2
    for i in range(len(b)):
        b[i] = cons_b*x_list[i]*(L-x_list[i])

    A = np.array(A)
    b = np.array(b)

    return A, b

def forward_sub(L, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i,:i], x[:i])) / L[i,i]
    return x

def backward_sub(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    return x

def gaussian_elimination(matrix, vector):
    # LU factorization
    [P, L, U] = lu(matrix)
    # solve for Lc=Pb
    Pb = np.dot(P, vector)
    c = forward_sub(L, Pb)
    # solve for Ux=c
    x = backward_sub(U, c)
    return x



def list_embed(n1, n2, *args):
    for lst in args:
        lst.insert(0, n1)
        lst.append(n2)

def plot_figure(x_axis, title, xlab, ylab, *args):
    for (num, lab) in args:
        plt.plot(x_axis, num, label=lab)

    # Add labels and legend
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()

    # Show plot
    plt.grid(True)
    #plt.show()
    plt.savefig(fr"C:\Users\13464\Desktop\M348\Group Project 2\{title}.png")
    plt.clf()
    print(title, 'done')



def iteration(k):
    n = int(2**(k+1))
    h = L/n
    # this list contains partitions: x_1 to x_{n-1}
    # notice that list indexing will not correspond to the actual x indexing
    x_list = [h+i*h for i in range(n-1)]

    A, b = set_matrix(n, h, x_list)

    # solution, true value, error
    approx_w = list(gaussian_elimination(A, b))
    true_w = [analy_sol(x_list[i]) for i in range(n-1)]
    error_list = [abs(approx_w[i]-true_w[i]) for i in range(n-1)]

    # add the first and last entry for each list
    list_embed(0, L, x_list)
    list_embed(0, 0, approx_w, true_w, error_list)

    # for debug
    '''print("gaussian elimilation \t analytical solution")
    for i in range(n-1):
        print(approx_w[i], "\t", true_w[i])
    print()'''

    # plot approx vs true solution
    if k==1 or k==2:
        title = f'True and Approximate Solution with k={k}'
        xlabel = 'Beam Length (in)'
        ylabel = 'Deflection (in)'
        plot_figure(x_list, title, xlabel, ylabel, \
                    (approx_w,'approximate'), (true_w,'true'))

    # plot error
    title = f'Errors with k={k}'
    xlabel = 'Beam Length (in)'
    ylabel = 'Error (in)'
    plot_figure(x_list, title, xlabel, ylabel, (error_list,'error'))

    # middle error
    mid_point = ceil(len(error_list)/2)
    mid_error_list.append(error_list[mid_point])
    h_squared.append(h**2)

    # condition number
    cond_num = cond(A)
    cond_list.append(cond_num)
    h_inv_squared.append(h**(-2))



mid_error_list = []
h_squared = []

cond_list = []
h_inv_squared = []

def list_make_log(*args):
    for lst in args:
        lst = [log(i) for i in lst]
    return lst



def main():
    K = 21
    for k in range(1,K):
        iteration(k)

    list_make_log(mid_error_list, h_squared, cond_list, h_inv_squared)

    title = "Middle Error"
    xlabel = 'log(E)'
    ylabel = 'log(h^2)'
    plot_figure(mid_error_list, title, xlabel, ylabel, (h_squared, None))

    title = "Condition Number"
    xlabel = 'log(KN)'
    ylabel = 'log(h^{-2})'
    plot_figure(cond_list, title, xlabel, ylabel, (h_inv_squared, None))



if __name__ == "__main__":
    main()
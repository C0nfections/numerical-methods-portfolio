from Decompose import Decompose
from Substitute import Substitute
import numpy as np
"""
This function will perform LU Decomposition on matrix A.
Parameters:
A: Matrix in Ax=b
b: column vector in Ax=b
n: 
tol: tolerance value
x: column vector in Ax=b
error: error
"""


"""
Factors A (with pivoting). If that worked, solve for solution. 
"""
def LUDecomp(A, b, n, tol, x, error):
    # permutation list and scaling list both of length n
    o, s = np.zeros(n, dtype=int), np.zeros(n, dtype=float)

    # error set to 0 to mean no error. er = -1 -> there is an error
    error[0] = 0
    Decompose(A, n, tol, o, s, error)
    if error[0] != -1:
        Substitute(A, o, n, b, x)
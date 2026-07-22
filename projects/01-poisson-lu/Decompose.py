from Pivot import Pivot
"""
Parameters for Decompose(A, n, tol, o, s, error):
A: matrix
n: dimension of (square) matrix A 
tol: tolerance
o: permutation vector
s: scaling vector
"""


def Decompose(A, n, tol, o, s, error):
    # initializing o and s
    for i in range(0, n):
        o[i] = i
        s[i] = abs(A[i, 0])  # assuming A is a 2D numpy array
        for j in range(1, n):
            if abs(A[i, j]) > s[i]:
                s[i] = abs(A[i, j]) # s[i] set to the max abs in row i

        # if a row is zero, then A is singular
        if s[i] == 0:
            error[0] = -1
            return

    # elimination with scaled partial pivoting
    for k in range(0, n-1):
        Pivot(A, o, s, n, k)

        if abs(A[o[k], k] / s[o[k]]) < tol:
            error[0] = -1
            print(A[o[k], k] / s[o[k]])
            return

        # elimination below pivot
        for i in range(k+1, n):
            factor = A[o[i], k] / A[o[k], k]
            A[o[i], k] = factor
            for j in range(k+1, n):
                A[o[i], j] = A[o[i], j] - factor * A[o[k], j]

    # checks the last pivot
    if abs(A[o[n-1], n-1] / s[o[n-1]]) < tol:
        error[0] = -1
        print(A[o[n-1], n-1] / s[o[n-1]])
        return


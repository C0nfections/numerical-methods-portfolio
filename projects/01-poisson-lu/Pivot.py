"""
Parameters for Pivot(A, o, s, n, k):
A: matrix
o: permutation vector
s: scaling vector
n: dimension of (square) matrix A
k: current column index where we are selecting pivot during Gauss elimination
"""

"""
Does scaled partial pivoting by choosing p in k..n-1 that maximizes 
| A[o[p], k] | / s[o[p]] and then swaps o[p] and o[k].

Computes element (column k / largest element in that row) and takes the max.
"""

def Pivot(A, o, s, n, k):
    p = k
    big = abs(A[o[k], k] / s[o[k]])
    for ii in range(k+1, n):
        dummy = abs(A[o[ii], k] / s[o[ii]])
        if dummy > big:
            big = dummy
            p = ii

    dummy = o[p]
    o[p] = o[k]
    o[k] = dummy
    # end pivot


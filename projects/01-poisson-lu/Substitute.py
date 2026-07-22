"""
Parameters for Substitute(A, o, n, b, x):
A: matrix
o: permutation vector
n: dimension of (square) matrix A
b: vector b in Ax=b
x: vector x in Ax=b
"""

"""
Forward substitution solves the lower-triangular system: Ly = Pb where:
- L is lower-triangular (in your LU storage, diagonal entries of L are 1)
- P represents the row swaps from pivoting (stored as the permutation vector o)
- b is the original right-hand side
- y is an intermediate vector we solve for first

Think row-by-row. Row i of L y = P b is:
L[i,0]*y0 + L[i,1]*y1 + ... + L[i,i]*yi = (P b)[i]

To solve for yi, isolate it:
L[i,i]*yi = (P b)[i] - (L[i,0]*y0 + L[i,1]*y1 + ... + L[i,i-1]*y_{i-1})

In summation notation:
L[i,i]*yi = (P b)[i] - sum_{j=0}^{i-1} L[i,j] * yj
So: yi = ((P b)[i] - sum_{j<i} L[i,j]*yj) / L[i,i]
"""

def Substitute(A, o, n, b, x):
    # forward substitution
    for i in range(1, n):  # start from the second row since row 1 is done
        sum = b[o[i]]  # start with (Pb)_i
        for j in range(0, i):
            sum = sum - A[o[i], j] * b[o[j]]  # subtract L_ij * y_j for j<i
        b[o[i]] = sum  # store y_i back into b
    # after forward sub, modified b is effectively holding y (in permuted storage)

    # backward substitution
    x[n - 1] = b[o[n - 1]] / A[o[n - 1], n - 1]
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum = sum + A[o[i], j] * x[j]
        x[i] = (b[o[i]] - sum) / A[o[i], i]
    # modifies x, the solution vector in place

from sympy import symbols, Matrix

#Define symbols
delta, mu_b, B_perp, B_par, gs, gv = symbols('delta mu_b B_perp B_par gs gv')

#Define matrix
matrix = Matrix([
    [-0.5*delta-0.5*mu_b*gv*B_perp+0.5*mu_b*gs*B_perp, 0.5*gs*mu_b*B_par],
    [0.5*gs*mu_b*B_par, 0.5*delta-0.5*mu_b*gv*B_perp-0.5*mu_b*gs*B_perp]
])

ho_matrix = -matrix

'''
matrix = Matrix([
    [-0.5*delta, 0.5*gs*mu_b*B_par],
    [0.5*gs*mu_b*B_par, 0.5*delta]
])
'''

# Display the matrix
print("Matrix:")
print(matrix)

# Compute the eigenvalues
eigenvalues = matrix.eigenvals()
print("Eigenvalues:")
print(eigenvalues)

# Compute the eigenvectors
eigenvectors = matrix.eigenvects()
print("Eigenvectors:")
print(eigenvectors)

# Display the matrix
print("Matrix:")
print(ho_matrix)

# Compute the eigenvalues
eigenvalues = ho_matrix.eigenvals()
print("Eigenvalues:")
print(eigenvalues)

# Compute the eigenvectors
eigenvectors = ho_matrix.eigenvects()
print("Eigenvectors:")
print(eigenvectors)

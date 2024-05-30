"""
    operator_to_vector(A)

Converts a matrix or array-like object `A` into a one-dimensional vector.

## Arguments
- `A`: The input matrix or array-like object.

## Returns
A one-dimensional vector representing the elements of `A`.

## Examples
```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> operator_to_vector(A)
4-element Vector{Int64}:
 1
 3
 2
 4
```
"""
operator_to_vector(A) = vec(A)

"""
    vector_to_operator(v, d=round(Int, sqrt(prod(size(v)))))

Reshapes a one-dimensional vector `v` into a matrix of size `d` by `d`.

## Arguments
- `v`: The input one-dimensional vector.
- `d`: (Optional) The desired size of the resulting matrix. If not provided, it is calculated as the rounded integer square root of the number of elements in `v`.

## Returns
A matrix of size `d` by `d` representing the elements of `v` reshaped accordingly.

## Examples
```julia
julia> v = [1, 3, 2, 4]
4-element Vector{Int64}:
 1
 3
 2
 4

julia> vector_to_operator(v)
2×2 Matrix{Int64}:
 1  2
 3  4
```
"""
vector_to_operator(v, d=round(Int, sqrt(prod(size(v))))) = reshape(v, d, d)

"""
    left_superop(A, d=size(A)[2])

Computes the Liouville space left superoperator representation of a matrix `A`.

## Arguments
- `A`: The input matrix.
- `d`: (Optional) The size of the identity matrix. If not provided, it is set to the number of columns in `A`.

## Returns
The left superoperator matrix obtained by performing a Kronecker product between the identity matrix and `A`.

## Examples
```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> left_superop(A)
4×4 Matrix{Int64}:
 1  2  0  0
 3  4  0  0
 0  0  1  2
 0  0  3  4
```
"""
left_superop(A, d=size(A)[2]) = kron(Matrix(I,d,d), A)

"""
    right_superop(A, d=size(A)[1])

Computes the Liouville space right superoperator representation of a matrix `A`.

## Arguments
- `A`: The input matrix.
- `d`: (Optional) The size of the identity matrix. If not provided, it is set to the number of rows in `A`.

## Returns
The right superoperator matrix obtained by performing a Kronecker product between the transpose of `A` and the identity matrix.

## Examples
```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> right_superop(A)
4×4 Matrix{Int64}:
 1  0  3  0
 0  1  0  3
 2  0  4  0
 0  2  0  4
```
"""
right_superop(A, d=size(A)[1]) = kron(transpose(A), Matrix(I,d,d))

"""
    commutator_superop(A)

Computes the Liouville space superoperator representation of the commutator with a matrix `A`.

## Arguments
- `A`: The input matrix.

## Returns
The commutator superoperator matrix obtained by subtracting the right superoperator of `A` from the left superoperator of `A`.

## Examples
```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> commutator_superop(A)
4×4 Matrix{Int64}:
  0   2  -3   0
  3   3   0  -3
 -2   0  -3   2
  0  -2   3   0
```
"""
commutator_superop(A) = left_superop(A) - right_superop(A)

"""
    anticommutator_superop(A)

Computes the Liouville space superoperator representation of the anti-commutator with a matrix `A`.

## Arguments
- `A`: The input matrix.

## Returns
The anti-commutator superoperator matrix obtained by adding the right superoperator of `A` to the left superoperator of `A`.

## Examples
```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> anticommutator_superop(A)
4×4 Matrix{Int64}:
 2  2  3  0
 3  5  0  3
 2  0  5  2
 0  2  3  8
```
"""
anticommutator_superop(A) = left_superop(A) + right_superop(A)

"""
    hamiltonian_evolution_superop(H, dt)

Computes the superoperator corresponding to the Hamiltonian evolution of a system governed by the Hamiltonian matrix `H` over a time step `dt`.

## Arguments
- `H`: The Hamiltonian matrix representing the dynamics of the system.
- `dt`: The time step for the evolution.

## Returns
The superoperator matrix representing the Hamiltonian evolution over the specified time step.

## Examples
```julia
julia> H = [1 1im; -1im -1]
2×2 Matrix{Complex{Int64}}:
 1+0im   0+1im
 0-1im  -1+0im

julia> hamiltonian_evolution_superop(H, 0.1)
4×4 Matrix{ComplexF64}:
   0.990066+0.0im           0.098672+0.00993351im    0.098672-0.00993351im    0.00993351+0.0im
  -0.098672-0.00993351im    0.970199+0.197344im     -0.00993351+0.0im         0.098672+0.00993351im
  -0.098672+0.00993351im   -0.00993351+0.0im         0.970199-0.197344im      0.098672-0.00993351im
   0.00993351+0.0im        -0.098672-0.00993351im   -0.098672+0.00993351im    0.990066+0.0im
```
"""
function hamiltonian_evolution_superop(H, dt)
    U = cis(-Hermitian(H)*dt)
    return left_superop(U)*right_superop(adjoint(U))
end
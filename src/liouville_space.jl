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
    vector_to_operator(v)
    vector_to_operator(v, d::Int)

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
vector_to_operator(v) = vector_to_operator(v, round(Int, sqrt(length(v))))
vector_to_operator(v, d::Int) = reshape(v, d, d)

"""
    left_superop(A)
    left_superop(A, d::Int)

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
left_superop(A) = left_superop(A, size(A,2))
left_superop(A, d::Int) = kron(I(d), A)

"""
    right_superop(A)
    right_superop(A, d::Int)

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
right_superop(A) = right_superop(A, size(A,1))
right_superop(A, d::Int) = kron(transpose(A), I(d))

"""
    left_right_superop(A, B)

Computes the Liouville space superoperator representation of the operation `A ⋅ B`.

## Arguments
- `A`: The left matrix.
- `B`: The right matrix.

## Returns
The superoperator obtained by performing a Kronecker product between the transpose of `A` and `B`.
```
"""
left_right_superop(A, B) = kron(transpose(B), A)

"""
    commutator_superop(A)

Computes the Liouville space superoperator representation of the commutator with a matrix `A`.

## Arguments
- `A`: The input matrix.

## Returns
The commutator superoperator matrix obtained by subtracting the right superoperator of `A` from the left superoperator of `A`.
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
```
"""
anticommutator_superop(A) = left_superop(A) + right_superop(A)

"""
    time_evolution_superop(H, t)

Computes the superoperator corresponding to the Hamiltonian evolution of a system governed by the Hamiltonian matrix `H` over a time `t`.

## Arguments
- `H`: The Hamiltonian matrix representing the dynamics of the system.
- `t`: The time step for the evolution.

## Returns
The superoperator matrix representing the Hamiltonian evolution over the specified time step.
```
"""
function time_evolution_superop(H, t)
    U = time_evolution_operator(H, t)
    return left_right_superop(U, adjoint(U))
end

time_evolution_operator(H, t) = cis(-Hermitian(H)*t)

"""
    hamiltonian_evolution_superop(H)

Computes the superoperator corresponding to the Hamiltonian contribution to the time
evolution given by the Hamiltonian matrix `H`
```math
L(\\cdot) = -i[H, \\cdot]
```

## Arguments
- `H`: The Hamiltonian matrix representing the dynamics of the system.

## Returns
The superoperator matrix representing the Hamiltonian evolution.
```
"""
hamiltonian_evolution_superop(H) = -im*commutator_superop(H)

"""
    dissipator_superop(L, M)
    dissipator_superop(L)

Computes the superoperator corresponding to the Lindblad dissipator
```math
D(\\cdot) = L \\cdot M^\\dagger - \\frac{1}{2} \\{M^\\dagger L, \\cdot\\}
```
If only `L` is provided, it is assumed that `M = L`.

## Arguments
- `L`: The jump operator.

## Returns
The superoperator matrix representing the Lindblad dissipator.
```
"""
function dissipator_superop(L, M)
    return left_right_superop(L, adjoint(M)) - anticommutator_superop(adjoint(M)*L)/2
end

dissipator_superop(L) = dissipator_superop(L, L)

"""
    lindbladian_superop(H, Ls, γs::AbstractVector)
    lindbladian_superop(H, Ls, γs::AbstractMatrix)
    lindbladian_superop(H)

Computes the Lindblad superoperator for the given Hamiltonian `H`, jump operators `Ls`,
and decay rates `γs`.
```math
L = -i[H, \\cdot] + \\sum_i γ_{n,m} (L_n \\cdot L_m^\\dagger - \\frac{1}{2} \\{L_m^\\dagger L_n, \\cdot\\})
```
If only `H` is provided, the Lindblad superoperator is equivalent to the free
Hamiltonian evolution superoperator.

## Arguments
- `H`: The Hamiltonian.
- `Ls`: The jump operators.
- `γs`: The decay rates.

## Returns
The superoperator matrix representing the Lindblad dissipator.
```
"""
function lindbladian_superop(H, Ls, γs::AbstractVector)
    if length(Ls) != length(γs)
        throw(ArgumentError("The number of jump operators and decay rates must be the same."))
    end

    if length(Ls) > 0 && axes(H) != axes(Ls[1])
        throw(ArgumentError("The Hamiltonian and jump operators must have the same dimensions."))
    end

    L = hamiltonian_evolution_superop(H)
    for i in eachindex(Ls)
        L += γs[i]*dissipator_superop(Ls[i])
    end

    return L
end

function lindbladian_superop(H, Ls, γs::AbstractMatrix)
    if size(γs, 1) != size(γs, 2)
        throw(ArgumentError("The decay rates matrix must be square."))
    end

    if size(γs, 1) != length(Ls)
        throw(ArgumentError("The number of jump operators and decay rates must be the same."))
    end

    L = hamiltonian_evolution_superop(H)
    for n in eachindex(axes(γs,1))
        for m in eachindex(axes(γs,2))
            L += γs[n,m]*dissipator_superop(Ls[n], Ls[m])
        end
    end

    return L
end

lindbladian_superop(H) = hamiltonian_evolution_superop(H)
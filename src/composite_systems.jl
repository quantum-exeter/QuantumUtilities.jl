"""
    tensor(A, B...)

Computes the tensor product of matrices or array-like objects `A` and `B`.

## Arguments
- `A`: The first matrix or array-like object.
- `B...`: Additional matrices or array-like objects to compute their tensor product with `A`.

## Returns
The tensor product of the input matrices or array-like objects.

## Examples
```julia
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> B = [5 6; 7 8]
2×2 Matrix{Int64}:
 5  6
 7  8

julia> C = [9 10; 11 12]
2×2 Matrix{Int64}:
 9   10
 11  12

julia> tensor(A, B)
4×4 Matrix{Int64}:
  5   6  10  12
  7   8  14  16
 15  18  20  24
 21  24  28  32

julia> tensor(A, B, C)
8×8 Matrix{Int64}:
  45   50   54   60   90  100  108  120
  55   60   66   72  110  120  132  144
  63   70   72   80  126  140  144  160
  77   84   88   96  154  168  176  192
 135  150  162  180  180  200  216  240
 165  180  198  216  220  240  264  288
 189  210  216  240  252  280  288  320
 231  252  264  288  308  336  352  384
```
"""
tensor = kron

"""
    partial_trace(v::AbstractVector, keep, dims)

Computes the partial trace of a pure state `v`.

## Arguments
- `v::AbstractVector`: The input pure state.
- `keep`: The indices of the subsystems to keep in the partial trace.
- `dims`: The dimensions of the subsystems.

## Returns
The partial trace of the input state.

## Examples
```julia
julia> v = [1, 0]
2-element Vector{Int64}:
 1
 0

julia> w = [1, 1]/sqrt(2)
2-element Vector{Float64}:
 0.7071067811865475
 0.7071067811865475

julia> vw = tensor(v,w)
4-element Vector{Float64}:
 0.7071067811865475
 0.7071067811865475
 0.0
 0.0

julia> partial_trace(vw, [2], [2, 2])
2×2 Matrix{Float64}:
 0.5  0.5
 0.5  0.5
```
"""
function partial_trace(v::AbstractVector, keep, dims)
    return partial_trace(v*v', keep, dims)
end

"""
    partial_trace(ρ::AbstractMatrix, keep, dims)

Computes the partial trace of a density matrix `ρ` by tracing out the specified subsystems.

## Arguments
- `ρ::AbstractMatrix`: The input density matrix.
- `keep`: The indices of the subsystems to keep in the partial trace.
- `dims`: The dimensions of the subsystems.

## Returns
The partial trace of the input density matrix.

## Examples
```julia
julia> A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
4×4 Matrix{Int64}:
  1   2   3   4
  5   6   7   8
  9  10  11  12
 13  14  15  16

julia> partial_trace(A, [2], [2, 2])
2×2 Matrix{Int64}:
 12  14
 20  22
```
"""
function partial_trace(ρ::AbstractMatrix, keep, dims)
    keep = sort(keep, rev=true) # sort the kept dimensions; the order of input keep is ignored; reverse is needed due to column-major ordering (see later)

    numdims = length(dims)
    keepdim = prod(dims[keep])
    tracedim = prod(dims) ÷ keepdim

    dims = reverse(dims) # kron is column-major so we need to reverse the order of the dimensions
    keep = numdims + 1 .- keep # invert the numbering of the kept dimensions (see above)
    traceout = setdiff(1:numdims, keep) # the dimensions to be traced out

    newdimorder = (keep..., traceout...) # permute the dimensions so that the traced out dimensions are at the end

    B = reshape(ρ, (dims..., dims...))
    C = PermutedDimsArray(B, (newdimorder..., (newdimorder .+ numdims)...))
    D = reshape(C, (keepdim, tracedim, keepdim, tracedim))

    R = similar(D, (keepdim, keepdim))
    _partial_trace!(R, D)
end

function _partial_trace!(ρout, ρin)
    for I in CartesianIndices(ρout)
        t = zero(eltype(ρin))
        @inbounds @simd for k in axes(ρin,2)
            K = CartesianIndex(I.I[1], k, I.I[2], k)
            t += ρin[K]
        end
        ρout[I] = t
    end
    ρout
end
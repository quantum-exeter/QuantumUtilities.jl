"""
    tensor(A, B...)

Computes the tensor product of matrices or array-like objects `A` and `B`.

## Arguments
- `A`: The first matrix or array-like object.
- `B...`: Additional matrices or array-like objects to compute their tensor product with `A`.

## Returns
The tensor product of the input matrices or array-like objects.
```
"""
tensor = kron

"""
    partial_trace(v::AbstractVector, trace_indices::Union{Int,Tuple{Vararg{Int}}}, dims::Tuple{Vararg{Int}})

Computes the partial trace of a pure state `v`.

## Arguments
- `v::AbstractVector`: The input pure state.
- `trace_indices::Union{Int,Tuple{Vararg{Int}}}`: The indices of the subsystems to trace over (can be a single integer or a tuple of integers).
- `dims:Tuple{Vararg{Int}}`: The dimensions of the subsystems.

## Returns
The partial trace of the input state.
```
"""
function partial_trace(v::AbstractVector, trace_indices::NTuple{M, Int}, dims::NTuple{N, Int}) where {M, N}
    return partial_trace(v*v', trace_indices, dims)
end

partial_trace(v::AbstractVector, trace_index::Int, dims::NTuple{N, Int}) where N = partial_trace(v, (trace_index,), dims)

"""
    partial_trace(ρ::AbstractMatrix, trace_indices::Union{Int,Tuple{Vararg{Int}}}, dims::Tuple{Vararg{Int}})

Computes the partial trace of a density matrix `ρ` by tracing out the specified subsystems.

## Arguments
- `ρ::AbstractMatrix`: The input density matrix.
- `trace_indices::Union{Int,Tuple{Vararg{Int}}}`: The indices of the subsystems to trace over (can be a single integer or a tuple of integers).
- `dims:::Tuple{Vararg{Int}}`: The dimensions of the subsystems.

## Returns
The partial trace of the input density matrix.
```
"""
function partial_trace(ρ::AbstractMatrix, trace_indices::NTuple{M, Int}, dims::NTuple{N, Int}) where {M, N}
    axes(ρ, 1) == axes(ρ, 2) || throw(DimensionMismatch("input matrix must be square, but got $(size(ρ))"))
    prod(dims) == size(ρ, 1) || throw(DimensionMismatch("input dimensions must match the size of the input matrix, but got $(dims) and $(size(ρ))"))
    1 ≤ minimum(trace_indices) && maximum(trace_indices) ≤ N || throw(ArgumentError("The trace indices must be between one and the number of dimensions, but got $(trace_indices)."))

    traceout_rev = N + 1 .- trace_indices # invert the numbering of the dimensions to be traced over (see below)
    dims_rev = reverse(dims) # kron is column-major so we need to reverse the order of the dimensions
    dims_out = ntuple(k -> k ∈ traceout_rev ? 1 : dims_rev[k], N)

    totaldim = prod(dims)
    tracedim = _prod_masked(dims_rev, traceout_rev)
    keepdim = totaldim ÷ tracedim

    B = reshape(ρ, (dims_rev..., dims_rev...))
    R = similar(ρ, (keepdim, keepdim))
    S = reshape(R, (dims_out..., dims_out...))

    traceout_axes = ntuple(k -> k ∈ traceout_rev ? axes(B, k) : Base.OneTo(1), N)
    traceout_indices = CartesianIndices(traceout_axes)
    _partial_trace!(S, B, traceout_indices)

    return R
end

partial_trace(ρ::AbstractMatrix, trace_index::Int, dims::NTuple{N, Int}) where N = partial_trace(ρ, (trace_index,), dims)

function _partial_trace!(S, B, trace_indices)
    for I in CartesianIndices(S)
        t = zero(eltype(B))
        @inbounds @simd for J in trace_indices
            t += B[max(I, CartesianIndex((J.I..., J.I...)))]
        end
        S[I] = t
    end

    return S
end

@inline _prod_masked(t::NTuple{N, Int}, k::NTuple{M, Int}) where {N, M} = t[k[1]]*_prod_masked(t, k[2:end])
@inline _prod_masked(t::NTuple{N, Int}, k::Tuple{Int}) where N = t[k[1]]
@inline _prod_masked(t::NTuple{N, Int}, k::Tuple{}) where N = 1
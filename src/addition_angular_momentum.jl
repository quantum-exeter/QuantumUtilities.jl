"""
    add(S1::SpinLength, S2::SpinLength)

Return a tuple of the different possible total angular momentum values resulting from the addition of two angular momenta `S1` and `S2`.

# Arguments
- `S1::SpinLength`: The first angular momentum.
- `S2::SpinLength`: The second angular momentum.

# Returns
- A tuple containing all possible values of total angular momentum.
"""
function add(S1::SpinLength{N,D}, S2::SpinLength{M,P}) where {N,D,M,P}
    isless(S1, S2) && return +(S2, S1)
    return _sum(S1, S2)
end

function _sum(::SpinInteger{N}, ::SpinInteger{M}) where {N,M}
    return ntuple(i -> SpinInteger(N - M + (i-1)), 2M + 1)
end

function _sum(::SpinInteger{N}, ::SpinHalfInteger{M}) where {N,M}
    return ntuple(i -> SpinHalfInteger(2N - M + 2(i-1)), M + 1)
end

function _sum(::SpinHalfInteger{N}, ::SpinInteger{M}) where {N,M}
    return ntuple(i -> SpinHalfInteger(N - 2M + 2(i-1)), 2M + 1)
end

function _sum(::SpinHalfInteger{N}, ::SpinHalfInteger{M}) where {N,M}
    return ntuple(i -> SpinInteger((N - M)÷2 + (i-1)), M + 1)
end
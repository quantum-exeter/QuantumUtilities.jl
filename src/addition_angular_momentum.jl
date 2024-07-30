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
"""
    struct SpinLength{N,D}

A type representing a spin length `N/D`.

Note that `N` and `D` must be such that the ratio `N/D` is either an integer or
a half-integer.
"""
struct SpinLength{N,D}
    function SpinLength{N,D}() where {N,D}
        isinteger(N) || throw(ArgumentError("The spin length numerator must be an integer."))
        isinteger(D) || throw(ArgumentError("The spin length denominator must be an integer."))
        n = convert(Int, N)
        d = convert(Int, D)
        r = n//d
        n, d = numerator(r), denominator(r)
        d == 1 || d == 2 || throw(ArgumentError("The spin length must be an integer or a half integer."))
        new{n,d}()
    end
end

"""
    struct SpinInteger{N}

A type representing an integer spin length `N`.
"""
const SpinInteger{N} = SpinLength{N, 1}

"""
    struct SpinHalfInteger{N}

A type representing a half-integer spin length `N/2`.
"""
const SpinHalfInteger{N} = SpinLength{N, 2}

"""
    SpinLength(N, D)
    SpinLength(S0::Int)
    SpinLength(S0::Rational)
    SpinLength(S0)
    SpinInteger(S0)
    SpinHalfInteger(S0)

Convenient constructors for spin length types.
"""
SpinLength(N, D) = SpinLength{N,D}()
SpinLength(S0::Int) = SpinLength(S0, 1)
SpinLength(S0::Rational) = SpinLength(numerator(S0), denominator(S0))
SpinLength(S0) = isinteger(S0) ? SpinLength(S0, 1) : isinteger(2S0) ? SpinLength(2S0, 2) : SpinLength(S0, 1)

SpinInteger(S0) = SpinInteger{S0}()
SpinHalfInteger(S0) = SpinHalfInteger{S0}()

"""
    SpinHalf
    SpinOne
    SpinTwo

Convenient named spin lengths of the respective cases.
"""
const SpinHalf = SpinLength(1, 2)
const SpinOne = SpinLength(1, 1)
const SpinTwo = SpinLength(2, 1)

Base.numerator(::SpinLength{N,D}) where {N,D} = N
Base.denominator(::SpinLength{N,D}) where {N,D} = D

Base.isless(::SpinLength{N,D}, ::SpinLength{M,P}) where {N,D,M,P} = N*P < M*D

Base.convert(::Type{SpinLength}, n::Int) = SpinLength(n)
Base.convert(::Type{SpinLength}, r::Rational) = SpinLength(r)
Base.convert(::Type{Int}, ::SpinInteger{N}) where {N} = N
Base.convert(::Type{Rational}, ::SpinLength{N,D}) where {N,D} = N//D

Base.isreal(::SpinLength) = true
Base.isinteger(::SpinInteger) = true
Base.isinteger(::SpinHalfInteger) = false

Base.eltype(::SpinInteger) = Int
Base.eltype(::SpinHalfInteger) = Rational{Int}

Base.length(::SpinLength) = 1

Base.size(::SpinLength) = ()

Base.broadcastable(S0::SpinLength) = Ref(S0)

Base.iterate(S0::SpinLength) = (S0, nothing)
Base.iterate(::SpinLength, ::Any) = nothing

function Base.show(io::IO, S0::SpinInteger)
    print(io, "Spin-", numerator(S0))
end

function Base.show(io::IO, S0::SpinLength)
    print(io, "Spin-", numerator(S0), "/", denominator(S0))
end

"""
    spin_projections(S0::SpinLength; rev=false)

Return an iterator over the projections of a spin of length `S0`.
By default, the iterator ranges from smallest to largest projection.
The optional argument `rev` can be used to reverse the order (i.e. from largest to smallest).

# Arguments
- `S0::SpinLength`: The spin length.
- `rev=false`: Whether to reverse the order of the projections.

# Returns
- The iterator over the spin projections.
"""
spin_projections(S0::SpinLength; rev=false) = rev ? Iterators.reverse(_spin_projections(S0)) : _spin_projections(S0)

_spin_projections(::SpinInteger{N}) where {N} = -N:N
_spin_projections(::SpinHalfInteger{N}) where {N} = -N//2:N//2
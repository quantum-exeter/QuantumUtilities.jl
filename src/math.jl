"""
    usinc(x)

Computes the unnormalized sinc function value for the input `x`, defined as `sin(x)/x`.

## Arguments
- `x`: The input value.

## Returns
The unnormalized sinc function value of `x`.

## Examples
```julia
julia> usinc(0.5)
0.958851077208406

julia> usinc(0.0)
1.0
```
"""
usinc(x) = Base.sinc(x/π)

"""
    realifclose(x::Complex{T}; tol=eps(T)) where T <: AbstractFloat

Returns the real part of the complex number `x` if the imaginary part is close to zero within a specified tolerance, and returns `x` otherwise.

## Arguments
- `x::Complex{T}`: The input complex number.
- `tol::AbstractFloat`: The tolerance for considering the imaginary part close to zero. Defaults to `eps(T)`.

## Returns
The real part of `x` if the imaginary part is close to zero within the specified tolerance, otherwise `x` itself.

## Examples
```julia
julia> realifclose(2 + 0im)
2

julia> realifclose(1e-21 + 1e-21im)
1.0e-21

julia> realifclose(1e-21 + 1e-21im; tol=1e-22)
1.0e-21 + 1.0e-21im
```
"""
realifclose(x::Number; tol=nothing) = x
function realifclose(x::Complex{T}; tol=eps(T)) where T <: AbstractFloat
    abs(imag(x)) < tol ? real(x) : x
end

scrap(x::Number; tol=nothing) = x

"""
    scrap(x::AbstractFloat; tol=eps(typeof(x)))

Returns a floating-point number with a small value set to zero within a specified tolerance.

## Arguments
- `x::AbstractFloat`: The input floating-point number.
- `tol`: The tolerance for considering the input value close to zero. Defaults to `eps(typeof(x))`.

## Returns
A floating-point number with a small value set to zero within the specified tolerance.

## Examples
```julia
julia> scrap(1.2e-21)
0.0

julia> scrap(1.2e-21; tol=1e-22)
1.2e-21
```
"""
scrap(x::AbstractFloat; tol=eps(typeof(x))) = abs(x) < tol ? zero(x) : x

"""
    scrap(x::Complex{T}; tol=eps(T)) where T <: AbstractFloat

Returns a complex number with small real and imaginary parts set to zero within a specified tolerance.

## Arguments
- `x::Complex{T}`: The input complex number.
- `tol::AbstractFloat`: The tolerance for considering the real and imaginary parts close to zero. Defaults to `eps(T)`.

## Returns
A complex number with small real and imaginary parts set to zero within the specified tolerance.

## Examples
```julia
julia> scrap(1e-20 + 1e-21im)
1.0e-20 + 1.0e-21im

julia> scrap(1e-20 + 1e-25im)
1.0e-20
```
"""
function scrap(x::Complex{T}; tol=eps(T)) where T <: AbstractFloat
    (abs(real(x)) < tol ? zero(T) : real(x)) + (abs(imag(x)) < tol ? zero(T) : imag(x))*im
end

"""
    cauchy_quadgk(g, a, b; kws...)

Computes the Cauchy principal value of the integral of a function `g` over the interval `[a, b]` using the `quadgk` quadrature method.

## Arguments
- `g`: The function to integrate.
- `a`: The lower bound of the interval.
- `b`: The upper bound of the interval.
- `kws...`: Additional keyword arguments accepted by `quadgk`.

## Returns
A tuple `(I, E)` containing the approximated integral `I` and an estimated upper bound on the absolute error `E`.

## Throws
- `ArgumentError`: If the interval `[a, b]` does not include zero.

## Examples
```julia
julia> g(x) = 1 / (x^2 + 1)

julia> cauchy_quadgk(g, -1, 1)
(-1.1080229582878788e-15, 1.312923433975867e-16)
```
"""
function cauchy_quadgk(g, a, b; kws...)
    a < zero(a) < b || throw(ArgumentError("domain must include 0"))
    g₀ = g(zero(a))
    g₀int = b == -a ? zero(g₀) : g₀ * log(abs(b/a)) / (b - a)
    return quadgk(x -> (g(x)-g₀)/x + g₀int, a, 0, b; kws...)
end
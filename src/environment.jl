"""
    abstract type EnvironmentType

Abstract type to represent the different kinds of environments (i.e. bosonic or fermionic).
"""
abstract type EnvironmentType end

"""
    struct BosonicType

Type representing the bosonic nature of an environment.
"""
struct BosonicType <: EnvironmentType end

Base.show(io::IO, ::BosonicType) = print(io, "Bosonic environment")

"""
    struct FermionicType

Type representing the fermionic nature of an environment.
"""
struct FermionicType <: EnvironmentType end

Base.show(io::IO, ::FermionicType) = print(io, "Fermionic environment")

"""
    struct Environment

Type representing an environment.
"""
struct Environment{E<:EnvironmentType, S<:AbstractSD, T}
    J::S
    kT::T
    μ::T

    function Environment{E, S, T}(J, kT, μ=zero(T)) where {E, S, T}
        new(J, kT, μ)
    end
end

"""
    Environment{E}(J, kT)

Construct an environment of type `E`, with spectral density `J` and temperature `kT`.

# Arguments
- `J`: The spectral density. Must be a subtype of `AbstractSD`.
- `kT`: The temperature (times the Boltzmann constant).

# Returns
- The environment.
"""
function Environment{E}(J::S, kT::T) where {E, S, T}
    Environment{E, S, T}(J, kT)
end

"""
    Environment{E}(J, kT, μ)

Construct an environment of type `E`, with spectral density `J`, temperature `kT`, and chemical potential `μ`.

# Arguments
- `J`: The spectral density. Must be a subtype of `AbstractSD`.
- `kT`: The temperature (times the Boltzmann constant).
- `μ`: The chemical potential.

# Returns
- The environment.
"""
function Environment{E}(J::S, kT::T1, μ::T2) where {E, S, T1, T2}
    T = promote_type(T1, T2)
    Environment{E, S, T}(J, T(kT), T(μ))
end

"""
    Environment(E, J, kT)

Construct an environment of type `E`, with spectral density `J` and temperature `kT`.

# Arguments
- `E`: Type of the environment. Must be as subtype of `EnvironmentType`.
- `J`: The spectral density. Must be a subtype of `AbstractSD`.
- `kT`: The temperature (times the Boltzmann constant).

# Returns
- The environment.
"""
function Environment(::Type{E}, J::S, kT::T) where {E<:EnvironmentType, S, T}
    Environment{E}{S, T}(J, kT)
end

"""
    Environment(E, J, kT, μ)

Construct an environment of type `E`, with spectral density `J`, temperature `kT`, and chemical potential `μ`.

# Arguments
- `E`: Type of the environment. Must be as subtype of `EnvironmentType`.
- `J`: The spectral density. Must be a subtype of `AbstractSD`.
- `kT`: The temperature (times the Boltzmann constant).
- `μ`: The chemical potential.

# Returns
- The environment.
"""
function Environment(::Type{E}, J::S, kT::T1, μ::T2) where {E<:EnvironmentType, S, T1, T2}
    T = promote_type(T1, T2)
    Environment{E}{S, T}(J, T(kT), T(μ))
end

"""
    struct BosonicEnvironment{S, T}

Type representing a bosonic environment. It is just an alias for `Environment{BosonicType, S, T}`.
"""
const BosonicEnvironment{S, T} = Environment{BosonicType, S, T}

"""
    struct FermionicEnvironment{S, T}

Type representing a fermionic environment. It is just an alias for `Environment{FermionicType, S, T}`.
"""
const FermionicEnvironment{S, T} = Environment{FermionicType, S, T}

"""
    BosonicEnvironment(J, kT)

Construct a bosonic environment with spectral density `J` and temperature `kT`.

# Arguments
- `J`: The spectral density. Must be a subtype of `AbstractSD`.
- `kT`: The temperature (times the Boltzmann constant).

# Returns
- The bosonic environment.
"""
BosonicEnvironment(J, kT) = Environment(BosonicType, J, kT)

"""
    FermionicEnvironment(J, kT, μ)

Construct a bosonic environment with spectral density `J`, temperature `kT`, and chemical potential `μ`.

# Arguments
- `J`: The spectral density. Must be a subtype of `AbstractSD`.
- `kT`: The temperature (times the Boltzmann constant).
- `μ`: The chemical potential.

# Returns
- The fermionic environment.
"""
FermionicEnvironment(J, kT, μ) = Environment(FermionicType, J, kT, μ)

function Base.show(io::IO, b::BosonicEnvironment)
    print(io, "Bosonic environment: kT = $(b.kT), J = $(b.J)")
end

function Base.show(io::IO, f::FermionicEnvironment)
    print(io, "Fermionic environment: kT = $(f.kT), μ = $(f.μ), J = $(f.J)")
end

Base.broadcastable(e::Environment) = Ref(e)

"""
    occupation(env::BosonicEnvironment, ω)

Return the occupation number of the bosonic environment `env` at frequency `ω`, that is
```math
n_\\mathrm{B}(\\omega) = \\frac{1}{e^{\\frac{\\omega}{\\mathrm{kT}}} - 1},
```
where `\\mathrm{kT}` is the temperature of `env`.

# Arguments
- `env`: The bosonic environment.
- `ω`: The frequency.

# Returns
- The bosonic occupation at the given frequency.
"""
occupation(env::BosonicEnvironment, ω) = (coth(ω/(2env.kT)) - 1)/2

"""
    occupation(env::FermionicEnvironment, ω)

Return the occupation number of the fermionic environment `env` at frequency `ω`, that is
```math
n_\\mathrm{F}(\\omega) = \\frac{1}{e^{\\frac{\\omega-\\mu}{\\mathrm{kT}}} + 1},
```
where `\\mathrm{kT}` is the temperature and `\\mu` the chemical potential of `env`.

# Arguments
- `env`: The fermionic environment.
- `ω`: The frequency.

# Returns
- The fermionic occupation at the given frequency.
"""
occupation(env::FermionicEnvironment, ω) = (1 - tanh((ω-env.μ)/(2env.kT)))/2
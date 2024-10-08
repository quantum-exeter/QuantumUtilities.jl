module QuantumUtilities

using LinearAlgebra
using BandedMatrices
using SpecialFunctions
using LogExpFunctions
using ClassicalOrthogonalPolynomials
using QuadGK
using SpectralDensities

include("composite_systems.jl")

export tensor, partial_trace

include("liouville_space.jl")

export operator_to_vector, vector_to_operator,
       left_superop, right_superop, left_right_superop,
       commutator_superop, anticommutator_superop,
       time_evolution_superop, time_evolution_operator,
       hamiltonian_evolution_superop, dissipator_superop,
       lindbladian_superop

include("math.jl")

export usinc, realifclose, scrap, cauchy_quadgk

include("harmonic_oscillator.jl")

export number_operator, position_operator, momentum_operator,
       creation_operator, annihilation_operator, coherent_state, displacement_operator

include("spinlength.jl")
include("addition_angular_momentum.jl")

export SpinLength, SpinInteger, SpinHalfInteger, SpinHalf, SpinOne, SpinTwo,
       spin_projections, add

include("spins.jl")

export sz_operator, sx_operator, sy_operator, sp_operator, sm_operator,
       s2_operator, rotation_operator

include("environment.jl")

export EnvironmentType, BosonicType, FermionicType
export Environment, BosonicEnvironment, FermionicEnvironment, occupation

end

module QuantumUtilities

using LinearAlgebra
using BandedMatrices
using SpecialFunctions
using LogExpFunctions
using ClassicalOrthogonalPolynomials
using QuadGK

include("composite_systems.jl")

export tensor, partial_trace

include("liouville_space.jl")

export operator_to_vector, vector_to_operator,
       left_superop, right_superop, left_right_superop,
       commutator_superop, anticommutator_superop, hamiltonian_evolution_superop

include("math.jl")

export usinc, realifclose, scrap, cauchy_quadgk

include("harmonic_oscillator.jl")

export number_operator, position_operator, momentum_operator,
       creation_operator, annihilation_operator, coherent_state, displacement_operator

include("spins.jl")

export SpinLength, SpinHalf, SpinOne, sz_operator, sx_operator, sy_operator,
       sp_operator, sm_operator, s2_operator, rotation_operator

end

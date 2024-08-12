using QuadGK
using QuantumUtilities
using Test

@testset "QuantumUtilities.jl" begin
    include("test_composite_systems.jl")

    include("test_liouville_space.jl")

    include("test_math_utils.jl")

    include("test_harmonic_oscillator.jl")

    include("test_spins.jl")

    include("test_environment.jl")
end

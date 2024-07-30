using QuantumUtilities
using LinearAlgebra

@testset "Spin length" begin
    @test SpinLength(3, 2) == SpinLength(3//2)
    @test SpinLength(3/2) == SpinLength(3//2)
    @test SpinLength(3) == SpinLength(3//1)
    @test SpinInteger(5) == SpinLength(5//1)
    @test SpinHalfInteger(5) == SpinLength(5//2)
    @test_throws ArgumentError SpinLength(3//4)
    @test_throws ArgumentError SpinLength(1.2)

    @test eltype(spin_projections(SpinHalf)) <: Rational
    @test eltype(spin_projections(SpinOne)) <: Integer

    @test length(spin_projections(SpinHalf)) == 2
    @test length(spin_projections(SpinOne)) == 3
    @test length(spin_projections(SpinLength(5//2))) == 6
    @test length(spin_projections(SpinInteger(5))) == 11

    sone_ms = spin_projections(SpinOne)
    sone_ms_rev = spin_projections(SpinOne; rev=true)
    @test sone_ms[1] == -1
    @test sone_ms[end] == 1
    @test sone_ms[1] == sone_ms_rev[end]

    shalf_ms = spin_projections(SpinHalf)
    shalf_ms_rev = spin_projections(SpinHalf; rev=true)
    @test shalf_ms[1] == -1//2
    @test shalf_ms[end] == 1//2
    @test shalf_ms[1] == shalf_ms_rev[end]

    @test sum(length.(spin_projections.(add(SpinHalf, SpinHalf)))) == length(shalf_ms)*length(shalf_ms)
    @test sum(length.(spin_projections.(add(SpinOne, SpinHalf)))) == length(sone_ms)*length(shalf_ms)
end

@testset "Spin operators" begin
    for S0 in [SpinLength(1//2), SpinLength(1), SpinLength(3//2), SpinLength(2)]
        Sz = sz_operator(S0)
        Sx = sx_operator(S0)
        Sy = sy_operator(S0)
        Sp = sp_operator(S0)
        Sm = sm_operator(S0)
        S2 = s2_operator(S0)

        @test Sp ≈ Sx + 1im*Sy
        @test Sm ≈ Sx - 1im*Sy
        @test S2 ≈ Sz^2 + Sx^2 + Sy^2
        @test (Sx*Sy - Sy*Sx) ≈ 1im*Sz
        @test (Sy*Sz - Sz*Sy) ≈ 1im*Sx
        @test (Sz*Sx - Sx*Sz) ≈ 1im*Sy
        @test (Sz*Sp - Sp*Sz) ≈ Sp
        @test (Sz*Sm - Sm*Sz) ≈ -Sm
        @test (Sp*Sm - Sm*Sp) ≈ 2*Sz
    end
end

@testset "Rotation operators" begin
    n = normalize!(rand(3))
    α = 2π*rand()

    @test rotation_operator(SpinHalf, n, α) ≈ QuantumUtilities.rotation_operator_generic(SpinHalf, n, α)
    @test rotation_operator(SpinOne, n, α) ≈ QuantumUtilities.rotation_operator_generic(SpinOne, n, α)

    α = 2π*rand()
    θ = π*rand()
    ϕ = 2π*rand()
    n = [sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ)]
    @test rotation_operator(SpinLength(3//2), θ, ϕ, α) ≈ rotation_operator(SpinLength(3//2), n, α)
end
using QuantumUtilities
using LinearAlgebra

@testset "Harmonic Oscillator" begin
    nc = 18
    α = 0.5 + 0.3im
    N = number_operator(nc)
    X = position_operator(nc)
    P = momentum_operator(nc)
    adag = creation_operator(nc)
    a = annihilation_operator(nc)
    ketα = coherent_state(α, nc)
    Dα = displacement_operator(α, nc)

    @test (X*P - P*X)[1:nc,1:nc] ≈ 1im*I
    @test (X^2/2 + P^2/2)[1:nc,1:nc] ≈ N[1:nc,1:nc] + (1/2)*I
    @test adjoint(a) ≈ adag
    @test (a + adag)/sqrt(2) ≈ X
    @test 1im*(adag - a)/sqrt(2) ≈ P
    @test adag*a ≈ N
    @test norm(ketα) ≈ 1
    @test (a*ketα)[1:nc] ≈ α*ketα[1:nc]
    @test Dα'*Dα ≈ I
    @test Dα*Dα' ≈ I

    nc, no = 42, 2
    α = 1.8 + 3.9im
    N = number_operator(nc, no)
    X = position_operator(nc, no)
    P = momentum_operator(nc, no)
    adag = creation_operator(nc, no)
    a = annihilation_operator(nc, no)
    ketα = coherent_state(α, nc, no)
    Dα = displacement_operator(α, nc)

    @test (X*P - P*X)[2:nc-no,2:nc-no] ≈ 1im*I
    @test (X^2/2 + P^2/2)[2:nc-no,2:nc-no] ≈ N[2:nc-no,2:nc-no] + (1/2)*I
    @test adjoint(a) ≈ adag
    @test (a + adag)/sqrt(2) ≈ X
    @test 1im*(adag - a)/sqrt(2) ≈ P
    @test (adag*a)[2:end] ≈ N[2:end]
    @test norm(ketα) ≈ 1
    @test (a*ketα)[2:nc-no] ≈ α*ketα[2:nc-no] skip=true
    @test Dα'*Dα ≈ I
    @test Dα*Dα' ≈ I

    @test coherent_state(0, 8) ≈ [1, 0, 0, 0, 0, 0, 0, 0, 0]
end
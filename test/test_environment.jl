using QuantumUtilities
using SpectralDensities

@testset "Environment" begin
    J = LorentzianSD(rand(), rand(), rand())
    kT = rand()
    μ = rand()

    b1 = Environment{BosonicType}(J, kT)
    b2 = BosonicEnvironment(J, kT)
    @test b1 === b2

    f1 = Environment{FermionicType}(J, kT, μ)
    f2 = FermionicEnvironment(J, kT, μ)
    @test f1 === f2

    nbos(ω) = 1/(exp(ω/kT) - 1)
    nfer(ω) = 1/(exp((ω-μ)/kT) + 1)
    ω = 0:0.1:10
    @test nbos.(ω) ≈ occupation.(b1, ω)
    @test nfer.(ω) ≈ occupation.(f1, ω)
end
using QuantumUtilities
using QuadGK

@testset "Math utilities" begin
    @testset "usinc" begin
        @test usinc(0.5) ≈ 0.958851077208406
        @test usinc(0.0) ≈ 1.0
    end

    @testset "realifclose" begin
        @test realifclose(2 + 0im) ≈ 2
        @test realifclose(1e-21 + 1e-21im) ≈ 1.0e-21
        @test realifclose(1e-21 + 1e-21im; tol=1e-22) ≈ 1.0e-21 + 1.0e-21im
    end

    @testset "scrap" begin
        @test scrap(1.2e-21) ≈ 0.0
        @test scrap(1.2e-21; tol=1e-22) ≈ 1.2e-21
        @test scrap(1e-10 + 1e-21im) ≈ 1.0e-10 + 1.0e-21im
        @test scrap(1e-10 + 1e-25im) ≈ 1.0e-10
    end

    @testset "cauchy_quadgk" begin
        g(x) = 1/(x+1)
        @test cauchy_quadgk(g, -1/2, 1/2)[1] ≈ -log(3)
    end
end
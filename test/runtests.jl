using LinearAlgebra
using QuadGK
using QuantumUtilities
using Test

@testset "QuantumUtilities.jl" begin
    @testset "Composite systems" begin
        v = rand(3)
        w = rand(4)
        v = v/norm(v)
        w = w/norm(w)
        vw = tensor(v,w)

        @test partial_trace(vw, [1], [3,4]) ≈ v*v'
        @test partial_trace(vw, [2], [3,4]) ≈ w*w'

        A = rand(2,2)
        B = rand(3,3)
        C = rand(5,5)
        D = rand(4,4)
        A = A/tr(A)
        B = B/tr(B)
        C = C/tr(C)
        D = D/tr(D)
        ABCD = tensor(A,B,C,D)

        @test partial_trace(ABCD, [1], [2,3,5,4]) ≈ A
        @test partial_trace(ABCD, [2], [2,3,5,4]) ≈ B
        @test partial_trace(ABCD, [3], [2,3,5,4]) ≈ C
        @test partial_trace(ABCD, [4], [2,3,5,4]) ≈ D

        @test partial_trace(ABCD, [1,2], [2,3,5,4]) ≈ tensor(A,B)
        @test partial_trace(ABCD, [1,3], [2,3,5,4]) ≈ tensor(A,C)
        @test partial_trace(ABCD, [1,4], [2,3,5,4]) ≈ tensor(A,D)
        @test partial_trace(ABCD, [2,3], [2,3,5,4]) ≈ tensor(B,C)
        @test partial_trace(ABCD, [3,4], [2,3,5,4]) ≈ tensor(C,D)

        @test partial_trace(ABCD, [1,2,3], [2,3,5,4]) ≈ tensor(A,B,C)
        @test partial_trace(ABCD, [1,2,4], [2,3,5,4]) ≈ tensor(A,B,D)
        @test partial_trace(ABCD, [1,3,4], [2,3,5,4]) ≈ tensor(A,C,D)
        @test partial_trace(ABCD, [2,3,4], [2,3,5,4]) ≈ tensor(B,C,D)
    end

    @testset "Liouville space" begin
        A = rand(7,7) + 1im*rand(7,7)
        B = rand(7,7) + 1im*rand(7,7)
        r = rand(7,7) + 1im*rand(7,7)

        vr = operator2vector(r)
        lA = LeftSuperOp(A)
        rB = RightSuperOp(B)

        @test vector2operator(lA*vr) ≈ A*r
        @test vector2operator(rB*vr) ≈ r*B

        @test vector2operator(CommutatorSuperOp(A)*vr) ≈ A*r - r*A
        @test vector2operator(AntiCommutatorSuperOp(A)*vr) ≈ A*r + r*A

        @test vector2operator(HamiltonianEvolutionSuperOp(B,1.0)*vr) ≈ exp(-1im*Hermitian(B))*r*adjoint(exp(-1im*Hermitian(B)))
    end

    @testset "Math utilities" begin
        @test usinc(0.5) ≈ 0.958851077208406
        @test usinc(0.0) ≈ 1.0

        @test realifclose(2 + 0im) ≈ 2
        @test realifclose(1e-21 + 1e-21im) ≈ 1.0e-21
        @test realifclose(1e-21 + 1e-21im; tol=1e-22) ≈ 1.0e-21 + 1.0e-21im

        @test scrap(1.2e-21) ≈ 0.0
        @test scrap(1.2e-21; tol=1e-22) ≈ 1.2e-21
        @test scrap(1e-10 + 1e-21im) ≈ 1.0e-10 + 1.0e-21im
        @test scrap(1e-10 + 1e-25im) ≈ 1.0e-10

        g(x) = 1/(x^2 + 1)
        @test cauchy_quadgk(g, -1, 1)[1] ≈ -1.1080229582878788e-15
    end
end

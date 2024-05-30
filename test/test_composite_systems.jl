using QuantumUtilities
using LinearAlgebra

@testset "Composite systems" begin
    v = rand(3)
    w = rand(4)
    v = v/norm(v)
    w = w/norm(w)
    vw = tensor(v,w)

    @test partial_trace(vw, 2, (3,4)) ≈ v*v'
    @test partial_trace(vw, 1, (3,4)) ≈ w*w'

    A = rand(2,2)
    B = rand(3,3)
    C = rand(5,5)
    D = rand(4,4)
    A = A/tr(A)
    B = B/tr(B)
    C = C/tr(C)
    D = D/tr(D)
    ABCD = tensor(A,B,C,D)

    @test partial_trace(ABCD, (2,3,4), (2,3,5,4)) ≈ A
    @test partial_trace(ABCD, (1,3,4), (2,3,5,4)) ≈ B
    @test partial_trace(ABCD, (1,2,4), (2,3,5,4)) ≈ C
    @test partial_trace(ABCD, (1,2,3), (2,3,5,4)) ≈ D

    @test partial_trace(ABCD, (3,4), (2,3,5,4)) ≈ tensor(A,B)
    @test partial_trace(ABCD, (2,4), (2,3,5,4)) ≈ tensor(A,C)
    @test partial_trace(ABCD, (2,3), (2,3,5,4)) ≈ tensor(A,D)
    @test partial_trace(ABCD, (1,4), (2,3,5,4)) ≈ tensor(B,C)
    @test partial_trace(ABCD, (1,2), (2,3,5,4)) ≈ tensor(C,D)

    @test partial_trace(ABCD, 4, (2,3,5,4)) ≈ tensor(A,B,C)
    @test partial_trace(ABCD, 3, (2,3,5,4)) ≈ tensor(A,B,D)
    @test partial_trace(ABCD, 2, (2,3,5,4)) ≈ tensor(A,C,D)
    @test partial_trace(ABCD, 1, (2,3,5,4)) ≈ tensor(B,C,D)

    @test partial_trace(ABCD, (3,4), (2,3,5,4)) ≈ partial_trace(ABCD, (4,3), (2,3,5,4))
    @test partial_trace(ABCD, (2,4), (2,3,5,4)) ≈ partial_trace(ABCD, (4,2), (2,3,5,4))
end
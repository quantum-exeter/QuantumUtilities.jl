using QuantumUtilities
using LinearAlgebra

@testset "Liouville space" begin
    A = rand(7,7) + 1im*rand(7,7)
    B = rand(7,7) + 1im*rand(7,7)
    C = rand(7,7) + 1im*rand(7,7)
    r = rand(7,7) + 1im*rand(7,7)

    vr = operator_to_vector(r)
    lA = left_superop(A)
    rB = right_superop(B)
    lrAB = left_right_superop(A, B)

    @test vector_to_operator(lA*vr) ≈ A*r
    @test vector_to_operator(rB*vr) ≈ r*B
    @test vector_to_operator(lrAB*vr) ≈ A*r*B

    @test vector_to_operator(commutator_superop(A)*vr) ≈ A*r - r*A
    @test vector_to_operator(anticommutator_superop(A)*vr) ≈ A*r + r*A

    @test vector_to_operator(time_evolution_superop(B,1.0)*vr) ≈ exp(-1im*Hermitian(B))*r*adjoint(exp(-1im*Hermitian(B)))
    @test vector_to_operator(hamiltonian_evolution_superop(B)*vr) ≈ -1im*(B*r - r*B)
    @test vector_to_operator(dissipator_superop(A,B)*vr) ≈ A*r*B' - 0.5*(B'*A*r + r*B'*A)

    @test lindbladian_superop(A, [], []) ≈ hamiltonian_evolution_superop(A)
    @test lindbladian_superop(zero(A), [B], [1]) ≈ dissipator_superop(B)
    @test lindbladian_superop(A, [B, C], [1.0, 0.5]) ≈ hamiltonian_evolution_superop(A) + dissipator_superop(B) + 0.5*dissipator_superop(C)
end
using QuantumUtilities
using LinearAlgebra

@testset "Spins" begin
    for S0 in [1//2, 1, 3//2, 2]
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
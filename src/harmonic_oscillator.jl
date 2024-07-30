"""
    number_operator(ncutoff::Int, noffset::Int=0)

Create the number operator matrix in the Fock basis for a quantum harmonic oscillator.

# Arguments
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the number operator.
"""
function number_operator(ncutoff::Int, noffset::Int=0)
    Diagonal(noffset:ncutoff)
end

"""
    position_operator(ncutoff::Int, noffset::Int=0)

Create the position operator matrix in the Fock basis for a quantum harmonic oscillator.

# Arguments
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the position operator.
"""
function position_operator(ncutoff::Int, noffset::Int=0)
    Hermitian(BandedMatrix(1 => [sqrt(n/2) for n in 1+noffset:ncutoff]), :U)
end
 
"""
    momentum_operator(ncutoff::Int, noffset::Int=0)

Create the momentum operator matrix in the Fock basis for a quantum harmonic oscillator.

# Arguments
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the momentum operator.
"""
function momentum_operator(ncutoff::Int, noffset::Int=0)
    Hermitian(BandedMatrix(1 => [-1im*sqrt(n/2) for n in 1+noffset:ncutoff]), :U)
end

"""
    creation_operator(ncutoff::Int, noffset::Int=0)

Create the creation (raising) operator matrix in the Focks basis for a quantum harmonic oscillator.

# Arguments
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the creation operator.
"""
function creation_operator(ncutoff::Int, noffset::Int=0)
    BandedMatrix(-1 => [sqrt(n+1) for n in noffset:ncutoff-1])
end

"""
    annihilation_operator(ncutoff::Int, noffset::Int=0)

Create the annihilation (lowering) operator matrix in the Fock basis for a quantum harmonic oscillator.

# Arguments
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the annihilation operator.
"""
function annihilation_operator(ncutoff::Int, noffset::Int=0)
    BandedMatrix(1 => [sqrt(n) for n in 1+noffset:ncutoff])
end

"""
    coherent_state(α, ncutoff::Int, noffset::Int=0)

Create a coherent state `α` in the Fock basis of a quantum harmonic oscillator.

The state is created by applying the displacement operator to the vacuum state:
```math
\\left|\\alpha\\right\\rangle = D(\\alpha)\\left|0\\right\\rangle.
```

# Arguments
- `α`: The complex amplitude of the coherent state.
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
-  The coherent state.
"""
function coherent_state(α, ncutoff::Int, noffset::Int=0)
    vacuum = zeros(ComplexF64, ncutoff+1-noffset)
    vacuum[1] = 1
    D = displacement_operator(α, ncutoff, noffset)
    return D*vacuum 
end

"""
    coherent_state_analytic(α, ncutoff::Int, noffset::Int=0)

Create a coherent state `α` in the Fock basis of a quantum harmonic oscillator.

The state is created by using the analytical expression for the state coefficients:
```math
\\left|\\alpha\\right\\rangle = \\sum_{n=\\mathrm{noffset}}^{\\mathrm{ncutoff}} e^{-\\frac{|\\alpha|^2}{2}}\\frac{\\alpha^n}{\\sqrt{n!}}\\left|n\\right\\rangle.
```
Due to the truncation of the series, the state is not guaranteed to be normalized.

# Arguments
- `α`: The complex amplitude of the coherent state.
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
-  The coherent state.
"""
function coherent_state_analytic(α, ncutoff::Int, noffset::Int=0)
    state = Vector{ComplexF64}(undef, ncutoff+1-noffset)

    for n in noffset:ncutoff
        state[n+1-noffset] = coherent_state_coefficient(α, n)
    end

    return state
end

"""
    coherent_state_coefficient(α, n::Int)

Coefficient in the Fock basis for the coherent state `α` of a quantum harmonic oscillator:
```math
\\left\\langle n\\middle|\\alpha\\right\\rangle = e^{-\\frac{|\\alpha|^2}{2}}\\frac{\\alpha^n}{\\sqrt{n!}}.
```

# Arguments
- `α`: The complex amplitude of the coherent state.
- `n::Int`: The Fock basis occupation number of the coefficient.

# Returns
-  The coherent state coefficient in the Fock basis.
"""
function coherent_state_coefficient(α, n::Int)
    exp(-0.5*α*α' + xlogy(n, complex(α)) - 0.5*logfactorial(n))
end

"""
    displacement_operator(α, ncutoff::Int, noffset::Int=0)

Create the displacement operator for a quantum harmonic oscillator.

The operator is constructed using the definition as unitary operator based on
the creation and annihilation operators:
```math
D(\\alpha) = e^{\\alpha a^\\dagger - \\alpha^* a}.
```

# Arguments
- `α`: The complex displacement parameter.
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the displacement operator.
"""
function displacement_operator(α, ncutoff::Int, noffset::Int=0)
    a = annihilation_operator(ncutoff, noffset)
    H = Hermitian(-1im*(α*a' - α'*a))
    D = cis(H)
    return D
end

"""
    displacement_operator_analytic(α, ncutoff::Int, noffset::Int=0)

Create the displacement operator for a quantum harmonic oscillator.

The operator is constructed using the analytic expressions of its matrix
elements in the Fock basis:
```math
\\left\\langle m \\middle| D(\\alpha) \\middle| n \\right\\rangle = 
e^{-\\frac{|\\alpha|^2}{2}} \\sqrt{\\frac{n!}{m!}} \\alpha^{m-n} L_n^{(m-n)}(|\\alpha|^2),
```
where ``L_n^{(m-n)}`` are the generalised Laguerre polynomial.

Due to the truncation of the matrix elements, the operator is not guaranteed to be unitary.

# Arguments
- `α`: The complex displacement parameter.
- `ncutoff::Int`: The cutoff value for the number of excitations to be included.
- `noffset::Int=0`: The offset for the number of excitations, defaults to 0.

# Returns
- The matrix representation of the displacement operator.
"""
function displacement_operator_analytic(α, ncutoff::Int, noffset::Int=0)
    D = Matrix{ComplexF64}(undef, (ncutoff+1-noffset, ncutoff+1-noffset))

    for m in noffset:ncutoff
        for n in noffset:ncutoff
            D[m+1-noffset, n+1-noffset] = displacement_operator_coefficient(α, m, n)
        end
    end

    return D
end

"""
    displacement_operator_coefficient(α, m::Int, n::Int)

Return the displacement operator matrix elements in the Fock basis for a quantum harmonic oscillator:
```math
\\left\\langle m \\middle| D(\\alpha) \\middle| n \\right\\rangle = 
e^{-\\frac{|\\alpha|^2}{2}} \\sqrt{\\frac{n!}{m!}} \\alpha^{m-n} L_n^{(m-n)}(|\\alpha|^2),
```
where `math L_n^{(m-n)}` are the generalised Laguerre polynomial.

# Arguments
- `α`: The complex displacement parameter.
- `m::Int`: The occupation number of the row.
- `n::Int`: The occupation number of the column.

# Returns
- The matrix elements of the displacement operator in the Fock basis.
"""
function displacement_operator_coefficient(α, m::Int, n::Int)
    if m ≥ n
        return exp(-0.5*α*α' + 0.5*logfactorial(n) - 0.5*logfactorial(m) + xlogy(m-n, complex(α)))*laguerrel(n, m-n, real(α*α'))
    else
        return conj(displacement_operator_coefficient(-α, n, m))
    end
end
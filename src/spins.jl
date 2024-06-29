const SpinLength = Union{Int, Rational}
const SpinHalf = 1//2
const SpinOne = 1

"""
    sz_operator(S0::SpinLength)

Create the matrix representation of the `z` spin component operator in the standard z basis.

# Arguments
- `S0::SpinLength`: The spin length.

# Returns
- The matrix representation of the ``S_z`` operator.
"""
function sz_operator(S0::SpinLength)
    Diagonal([m for m in S0:-1:-S0])
end
 
"""
    sx_operator(S0::SpinLength)

Create the matrix representation of the `x` spin component operator in the standard z basis.

# Arguments
- `S0::SpinLength`: The spin length.

# Returns
- The matrix representation of the ``S_x`` operator.
"""
function sx_operator(S0::SpinLength)
    Hermitian(BandedMatrix(1 => [sqrt(S0*(S0+1)-m*(m+1))/2 for m in S0-1:-1:-S0]), :U)
end
 
"""
    sy_operator(S0::SpinLength)

Create the matrix representation of the `y` spin component operator in the standard z basis.

# Arguments
- `S0::SpinLength`: The spin length.

# Returns
- The matrix representation of the ``S_y`` operator.
"""
function sy_operator(S0::SpinLength)
    Hermitian(BandedMatrix(1 => [sqrt(S0*(S0+1)-m*(m+1))/2im for m in S0-1:-1:-S0]), :U)
end

"""
    sp_operator(S0::SpinLength)

Create the matrix representation of the raising spin ladder operator in the standard z basis.

# Arguments
- `S0::SpinLength`: The spin length.

# Returns
- The matrix representation of the ``S_+`` operator.
"""
function sp_operator(S0::SpinLength)
    BandedMatrix(1 => [sqrt(S0*(S0+1)-m*(m+1)) for m in S0-1:-1:-S0])
end

"""
    sm_operator(S0::SpinLength)

Create the matrix representation of the lowering spin ladder operator in the standard z basis.

# Arguments
- `S0::SpinLength`: The spin length.

# Returns
- The matrix representation of the ``S_-`` operator.
"""
function sm_operator(S0::SpinLength)
    BandedMatrix(-1 => [sqrt(S0*(S0+1)-m*(m-1)) for m in S0:-1:-S0+1])
end

"""
    s2_operator(S0::SpinLength)

Create the matrix representation of the squared spin operator in the standard z basis.

# Arguments
- `S0::SpinLength`: The spin length.

# Returns
- The matrix representation of the ``S^2`` operator.
"""
function s2_operator(S0::SpinLength)
    Diagonal([S0*(S0+1) for m in S0:-1:-S0])
end

"""
    rotation_operator(S0::SpinLength, n, α)

Create the rotation operator matrix for a given spin length and rotation parameters.

# Arguments
- `S0::SpinLength`: The spin length.
- `n`: The axis of rotation (must be a three-component vector-like).
- `α`: The rotation angle.

# Returns
- The matrix representation of the rotation operator for spin `S0`.
"""
function rotation_operator(S0::SpinLength, n, α)
    if S0 == SpinHalf
        return rotation_operator_spinhalf(n, α)
    elseif S0 == SpinOne
        return rotation_operator_spinone(n, α)
    else
        return rotation_operator_generic(S0, n, α)
    end
end

"""
    rotation_operator(S0::SpinLength, θ, φ, α)

Create the rotation operator matrix for a given spin length and rotation parameters.

# Arguments
- `S0::SpinLength`: The spin length.
- `θ`: The polar angle characterising the direction of the rotation.
- `φ`: The azimuthal angle characterising the direction of the rotation.
- `α`: The rotation angle.

# Returns
- The matrix representation of the rotation operator for spin `S0`.
"""
function rotation_operator(S0::SpinLength, θ, φ, α)
    rotation_operator(S0::SpinLength, [sin(θ)*cos(φ), sin(θ)*sin(φ), cos(θ)], α)
end

function rotation_operator_generic(S0::SpinLength, n, α)
    cis(-(α*Hermitian(n[1]*sx_operator(S0) + n[2]*sy_operator(S0) + n[3]*sz_operator(S0))))
end

function rotation_operator_spinhalf(n, α)
    [cos(α/2) - 1im*n[3]*sin(α/2) -(n[2] + 1im*n[1])*sin(α/2) ;
     (n[2] - 1im*n[1])*sin(α/2)  cos(α/2) + 1im*n[3]*sin(α/2)]
end

function rotation_operator_spinone(n, α)
    Sn = n[1]*sx_operator(SpinOne) + n[2]*sy_operator(SpinOne) + n[3]*sz_operator(SpinOne)
    return I(3) - im*Sn*sin(α) + Sn*Sn*(cos(α) - 1)
end
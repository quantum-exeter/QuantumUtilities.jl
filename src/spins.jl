const SpinLength = Union{Int, Rational}

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
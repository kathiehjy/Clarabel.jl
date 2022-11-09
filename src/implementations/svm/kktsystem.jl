function kkt_update!(
    kktsystem::SvmKKTSystem{T},
    data::SvmProblemData{T},
    cones::CompositeCone{T}
) where {T}

    #update the linear solver with new cones
    #kkt returns the ldl factorisation results and store the factor in is_success
    is_success  = kktsolver_update!(kktsystem.kktsolver,cones)  
    
    return is_success
end


function kkt_solve_initial_point!(
    kktsystem::SvmKKTSystem{T},
    variables::SvmVariables{T},
    data::SvmProblemData{T},
    cones::CompositeCone{T}
) where{T}

    variables_unit_initialization!(variables, cones)
    
    return nothing

end


function kkt_solve!(
    kktsystem::SvmKKTSystem{T},
    lhs::SvmVariables{T},      # Δ values, store the updated values of steps
    data::SvmProblemData{T},   # Store the problem formulation to set up the coefficient matrix
    variables::SvmVariables{T},   # Value of system variable from previous iteration
    residuals::SvmResiduals{T}
    steptype::Symbol   #:affine or :combined
) where{T}
"""variable means the value at each iteration, not the size of the step
"""

    rλ2 = residuals.rλ2
    rλ1 = residuals.rλ1
    rξ  = residuals.rξ
    rw  = residuals.rw
    λ1  = variables.λ1
    λ2  = variables.λ2
    q   = variables.q
    ξ   = variables.ξ

    n = data.n
    N = data.N

    # construct y and s
    s = data.y # s is label vector
    x = data.x # feature matrix
    y = data.Y # pairwise y = s * x

    
    # Construct D, used repeatedly 
    Dξ = Diagonal(ξ)
    Dλ1 = Diagonal(λ1)
    Dλ2 = Diagonal(λ2)
    Dq = Diagonal(q)
    D = inv(inv(Dλ2)*Dq + inv(Dλ1)*Dξ)

    # Construct the linear coefficient of linear system
    I1 = Matrix(1.0I, n, n)     # The identity matrix added to the left top corner
    TL = transpose(y) * D * y   # LT -- stand for top left element
    BL = -transpose(s) * D * y  # LB -- bottom left element 
    TR = transpose(BL)
    BR = transpose(s) * D * s
    #Linear_system_coef = [I1+TL, TR;BL, BR]
    top = hcat(I1+TL, TR)
    bottom = hcat(BL, BR)
    Linear_system_coef = vcat(top, bottom)


    # Construct the rhs of the linear system   rhs = [Trhs; Brhs]
    Trhs = -rw + transpose(y)*D*(-rξ+ξ+inv(Dλ1)*Dξ*rλ1-q)
    Brhs = -rλ2 .- transpose(s)*D*(-rξ+ξ+inv(Dλ1)*Dξ*rλ1-q)
    rhs = vcat(Trhs, Brhs)
    result = inv(Linear_system_coef) * rhs      # result = [Δw; Δb]
    Δw = result[1:end-1]
    Δb = result[end]

    # Solve for Δλ2, Δξ, Δq, Δλ1 given Δw, Δb
    Δλ2 = D * (-rξ -y*Δw + ξ + inv(Dλ1)*Dξ*rλ1 - q + s.*Δb)
    Δλ1 = rλ1 - Δλ2
    Δq = -q - inv(Dλ2)*Dq*Δλ2
    Δξ = -ξ - inv(Dλ1)*Dξ*Δλ1
    
    # Update the step size
    lhs.λ1 = Δλ1
    lhs.λ2 = Δλ2
    lhs.q  = Δq
    lhs.ξ  = Δξ
    lhs.w  = Δw
    lhs.b  = Δb

    return

end

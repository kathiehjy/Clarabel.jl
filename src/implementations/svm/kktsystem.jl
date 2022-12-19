function kkt_update!(
    kktsystem::SvmKKTSystem{T},
    data::SvmProblemData{T},
    cones::CompositeCone{T}
) where {T}

    #update the linear solver with new cones
    #kkt returns the ldl factorisation results and store the factor in is_success
    return true
    is_success  = kktsolver_update!(kktsystem.kktsolver,cones)  
    
    return is_success
end


function kkt_solve_initial_point!(
    variables::SvmVariables{T},
) where{T}

    variables_unit_initialization!(variables)
    
    return nothing

end


function kkt_solve!(
    kktsystem::SvmKKTSystem{T},
    lhs::SvmVariables{T},      # Δ values, store the updated values of steps
    rhs::SvmVariables{T},
    data::SvmProblemData{T},   # Store the problem formulation to set up the coefficient matrix
    variables::SvmVariables{T},   # Value of system variable from previous iteration
    steptype::Symbol   #:affine or :combined
) where{T}
"""variable means the value at each iteration, not the value of previous iterative step
"""
    # kktsystem structure has pre-allocated memory for all the variables used
    kktsystem.λ1  = deepcopy(variables.λ1)
    kktsystem.λ2  = deepcopy(variables.λ2)
    kktsystem.q   = deepcopy(variables.q)
    kktsystem.ξ   = deepcopy(variables.ξ)

    # residual, i.e., rhs of the linear system changes for affine and combined steptype
    """Simplify this after pass in the step_rhs as an input to this function"""
    kktsystem.rλ2 = rhs.b
    kktsystem.rλ1 = deepcopy(rhs.λ1)
    kktsystem.rξ  = deepcopy(rhs.ξ)
    kktsystem.rw  = deepcopy(rhs.w)
    kktsystem.const1 = deepcopy(rhs.λ2)
    kktsystem.const2 = deepcopy(rhs.q)
    

    n = data.n
    N = data.N

    # construct data and feature 
    kktsystem.y = deepcopy(data.y) # s is label vector
    kktsystem.x = deepcopy(data.x) # feature matrix
    kktsystem.Y = deepcopy(data.Y) # pairwise y = s * x

    
    # Construct D, used repeatedly 
    """Pre-allocate memory for D doesn't speed things up, no need to use kktsystem.D"""
    Dξ = Diagonal(kktsystem.ξ)
    Dλ1 = Diagonal(kktsystem.λ1)
    Dλ2 = Diagonal(kktsystem.λ2)
    Dq = Diagonal(kktsystem.q)
    D = inv(inv(Dλ2)*Dq + inv(Dλ1)*Dξ)

    # Construct the linear coefficient of linear system
    # I1 = Matrix(1.0I, n, n)     # The identity matrix added to the left top corner
    # TL = transpose(kktsystem.Y) * D * kktsystem.Y   # LT -- stand for top left element
    # BL = -transpose(kktsystem.y) * D * kktsystem.Y  # LB -- bottom left element 
    # TR = transpose(BL)
    # BR = transpose(kktsystem.y) * D * kktsystem.y
    # #Linear_system_coef = [I1+TL, TR;BL, BR]
    # top = hcat(I1+TL, TR)
    # bottom = hcat(BL, BR)
    # Linear_system_coef = vcat(top, bottom)

    kktsystem.Linear_system_coef[1:n,1:n] = transpose(kktsystem.Y) * D * kktsystem.Y + kktsystem.I1   # LT -- stand for top left element
    kktsystem.Linear_system_coef[end,1:n] = -transpose(kktsystem.y) * D * kktsystem.Y  # LB -- bottom left element 
    kktsystem.Linear_system_coef[1:n,end] = transpose(kktsystem.Linear_system_coef[end,1:n])
    kktsystem.Linear_system_coef[end,end] = transpose(kktsystem.y) * D * kktsystem.y
    #Linear_system_coef = [I1+TL, TR;BL, BR]



    # Construct the rhs of the linear system   rhs = [Trhs; Brhs]
    kktsystem.reuse = -kktsystem.rξ+inv(Dλ1)*kktsystem.const2+inv(Dλ1)*Dξ*kktsystem.rλ1-inv(Dλ2)*kktsystem.const1
    # Trhs = -kktsystem.rw + transpose(kktsystem.Y)*D*kktsystem.reuse
    # Brhs = -kktsystem.rλ2 .- transpose(kktsystem.y)*D*kktsystem.reuse
    # kktsystem.Rhs = vcat(Trhs, Brhs) # dimension = n + 1
    kktsystem.Rhs[1:n] = -kktsystem.rw + transpose(kktsystem.Y)*D*kktsystem.reuse
    kktsystem.Rhs[end] = -kktsystem.rλ2 - transpose(kktsystem.y)*D*kktsystem.reuse

    Factor = lu(kktsystem.Linear_system_coef)
    result = Factor \ kktsystem.Rhs
    try 
        # result = (Linear_system_coef) \ kktsystem.Rhs      # result = [Δw; Δb]
        result = Factor \ kktsystem.Rhs
    catch
        println("D = ",D)
        println("BR = ",BR)
        println("λ1 = ",λ1)
        println("λ2 = ",λ2)
        println("q = ",q)
        println("ξ = ",ξ)

        error("Foo")
    end 
    # result = (Linear_system_coef) \ kktsystem.Rhs 
    result = Factor \ kktsystem.Rhs

    # lhs.w = Δw; lhs.b = Δb
    lhs.w .= result[1:end-1]
    lhs.b = result[end]
    constructVector = ones(N, 1)
    # Solve for Δλ2, Δξ, Δq, Δλ1 given Δw, Δb with pre-allocated memory
    # lhs.λ1 = Δλ1; lhs.λ2 = Δλ2; lhs.q = Δq; lhs.ξ = Δξ
    lhs.λ2 .= D * (-kktsystem.rξ -kktsystem.Y*lhs.w + inv(Dλ1)*kktsystem.const2 + inv(Dλ1)*Dξ*kktsystem.rλ1 - inv(Dλ2)*kktsystem.const1 + kktsystem.y*lhs.b)
    lhs.λ1 .= kktsystem.rλ1 - lhs.λ2

    #lhs.q .= -inv(Dλ2) * Dq * Dλ2 * constructVector - inv(Dλ2)*Dq*lhs.λ2
    #lhs.ξ .= -inv(Dλ1) * Dλ1 * Dξ * constructVector - inv(Dλ1)*Dξ*lhs.λ1
    lhs.q .= -inv(Dλ2) * kktsystem.const1 - inv(Dλ2)*Dq*lhs.λ2
    lhs.ξ .= -inv(Dλ1) * kktsystem.const2 - inv(Dλ1)*Dξ*lhs.λ1

    # print("const1: ")
    # println(Dq * Dλ2 * constructVector)
    # # Update the step size
    # @. lhs.λ1 = Δλ1
    # @. lhs.λ2 = Δλ2
    # @. lhs.q  = Δq
    # @. lhs.ξ  = Δξ
    # @. lhs.w  = Δw
    #    lhs.b  = Δb

    return true

end

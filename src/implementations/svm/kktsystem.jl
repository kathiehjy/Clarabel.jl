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
"""variable means the value at each iteration, not the size of the step
"""
    λ1  = variables.λ1
    λ2  = variables.λ2
    q   = variables.q
    ξ   = variables.ξ

    # residual, i.e., rhs of the linear system changes for affine and combined steptype
    """Simplify this after pass in the step_rhs as an input to this function"""
    rλ2 = rhs.b
    rλ1 = rhs.λ1
    rξ  = rhs.ξ
    rw  = rhs.w
    const1 = rhs.λ2
    const2 = rhs.q
    

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
    reuse = -rξ+inv(Dλ1)*const2+inv(Dλ1)*Dξ*rλ1-inv(Dλ2)*const1
    Trhs = -rw + transpose(y)*D*reuse
    Brhs = -rλ2 .- transpose(s)*D*reuse
    rhs = vcat(Trhs, Brhs)
    try 
        result = (Linear_system_coef) \ rhs      # result = [Δw; Δb]
    catch
        println("D = ",D)
        println("BR = ",BR)
        println("λ1 = ",λ1)
        println("λ2 = ",λ2)
        println("q = ",q)
        println("ξ = ",ξ)

        error("Foo")
    end 
    result = (Linear_system_coef) \ rhs 
    Δw = result[1:end-1]
    Δb = result[end]

    # Solve for Δλ2, Δξ, Δq, Δλ1 given Δw, Δb
    Δλ2 = D * (-rξ -y*Δw + ξ + inv(Dλ1)*Dξ*rλ1 - q + s.*Δb)
    Δλ1 = rλ1 - Δλ2
    Δq = -q - inv(Dλ2)*Dq*Δλ2
    Δξ = -ξ - inv(Dλ1)*Dξ*Δλ1
    
    # Update the step size
    @. lhs.λ1 = Δλ1
    @. lhs.λ2 = Δλ2
    @. lhs.q  = Δq
    @. lhs.ξ  = Δξ
    @. lhs.w  = Δw
       lhs.b  = Δb

    return true

end

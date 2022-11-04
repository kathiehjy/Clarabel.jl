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
    lhs::SvmVariables{T},
    rhs::SvmVariables{T},
    data::SvmProblemData{T},
    variables::SvmVariables{T},
    cones::CompositeCone{T},
    steptype::Symbol   #:affine or :combined
) where{T}
"""variable means the value at each iteration, not the size of the step?
"""
    (w,b) = (kktsystem.w, kktsystem.b)



    ## Once get Δw and Δb, use them to get all the other variables
    #solve for Δλ2.
    #-----------
    # Numerator first
    ξ   = workx
    @. ξ = variables.x / variables.τ

    P   = Symmetric(data.P)

    tau_num = rhs.τ - rhs.κ/variables.τ + dot(data.q,x1) + dot(data.b,z1) + 2*quad_form(ξ,P,x1)

    #offset ξ for the quadratic form in the denominator
    ξ_minus_x2    = ξ   #alias to ξ, same as workx
    @. ξ_minus_x2  -= x2

    tau_den  = variables.κ/variables.τ - dot(data.q,x2) - dot(data.b,z2)
    tau_den += quad_form(ξ_minus_x2,P,ξ_minus_x2) - quad_form(x2,P,x2)

    #solve for (Δx,Δz)
    #-----------
    lhs.τ  = tau_num/tau_den
    @. lhs.x = x1 + lhs.τ * x2
    @. lhs.z = z1 + lhs.τ * z2


    #solve for Δs
    #-------------
    # compute the linear term HₛΔz, where Hs = WᵀW for symmetric
    # cones and Hs = μH(z) for asymmetric cones
    mul_Hs!(cones,lhs.s,lhs.z,workz)
    @. lhs.s = -(lhs.s + Δs_const_term)

    #solve for Δκ
    #--------------
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    # we don't check the validity of anything
    # after the KKT solve, so just return is_success
    # without further validation
    return is_success

end

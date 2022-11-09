using LinearAlgebra

function variables_calc_mu(
    variables::SvmVariables{T},
) where {T}
  N = length(variables.λ1)
  μ = (variables.λ1 ⋅ variables.ξ + variables.λ2 ⋅ variables.q)/(2 * N)

  return μ
end


function variables_calc_step_length(
    variables::SvmVariables{T},
    step::SvmVariables{T},
    cones::CompositeCone{T},
    settings::Settings{T},
    steptype::Symbol
) where {T}
    α = one(T)

    # ξ,q,λ1,λ2 are constrainted to be positive, 
    # so there is not need to check for sign, like for τ and κ
    (αξ,αq) = step_length(cones, step.ξ, step.q, variables.ξ, variables.q, settings, α)
    α = min(αξ, αq)
    (αλ1,αλ2) = step_length(cones, step.λ1, step.λ2, variables.λ1, variables.λ2, settings, α)
    α = min(αλ1, αλ2)


    if(steptype == :combined)
        α *= settings.max_step_fraction
    end

    return α    
end


function variables_barrier(
    variables::SvmVariables{T},
    step::SvmVariables{T},
    α::T,
    cones::CompositeCone{T},
) where {T}

    error("Function not yet implemented")

    #return barrier function
    # Not required -- leave empty
end

function variables_copy_from(dest::SvmVariables{T},src::SvmVariables{T}) where {T}
    dest.w .= src.w
    dest.b  = src.b
    dest.ξ .= src.ξ
    dest.λ1.= src.λ1
    dest.λ2.= src.λ2
    dest.q .= srs.q
end

function variables_scale_cones!(
    variables::SvmVariables{T},
    cones::CompositeCone{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    error("Function not yet implemented")
end


function variables_add_step!(
    variables::SvmVariables{T},
    step::SvmVariables{T}, α::T
) where {T}

    @. variables.w  += α*step.w
    variables.b     += α*step.b
    @. variables.ξ  += α*step.ξ
    @. variables.q  += α*step.q
    @. variables.λ1 += α*step.λ1
    @. variables.λ2 += α*step.λ2

    return nothing
end


function variables_affine_step_rhs!(
    d::SvmVariables{T},             # rhs of Newton's equation
    r::SvmResiduals{T},             # residual
    variables::SvmVariables{T},     # Value of the problem variable at each iterate

) where{T}

    @. d.w     =  r.rw
    @. d.ξ     =  r.rξ
    @. d.λ1    =  r.rλ1
    d.b        =  r.rλ2
    d.λ2      .=  diagm(variables.λ2) * variables.q
    d.q       .=  diagm(variables.λ1) * variables.ξ
    


    return nothing
end


function variables_combined_step_rhs!(
    d::SvmVariables{T},             # rhs of Newton's equation
    r::SvmResiduals{T},             # residual
    variables::SvmVariables{T},     # Value of the problem variable at each iterate
    step::SvmVariables{T},          # affine direction term
    σ::T,
    μ::T
) where {T}
    N = length(d.q)
    @. d.w  = (one(T) - σ)*r.rw
    @. d.ξ  = (one(T) - σ)*r.rξ
    @. d.λ1 = (one(T) - σ)*r.rλ1
    d.b     = (one(T) - σ)*r.rλ2
    d.λ2   .= diagm(variables.λ2)*variables.q-ones(T,1,N)*σ*μ-diagm(step.q)*step.λ2
    d.q    .= diagm(variables.λ1)*variables.ξ-ones(T,1,N)*σ*μ-diagm(step.λ1)*step.ξ

end

# Calls shift_to_cone on all conic variables and does not
# touch the primal variables. Used for symmetric problems.

function variables_symmetric_initialization!(
    variables::SvmVariables{T},
    cones::CompositeCone{T}
) where {T}
    """Leave empty
    """
    error("Function not yet implemented")
end


# Calls unit initialization on all conic variables and zeros
# the primal variables.   Used for nonsymmetric problems.
function variables_unit_initialization!(
    Data::SvmProblemData{T}
    variables::SvmVariables{T},
) where {T}

    variables.w .= zeros(T,1,Data.n)
    variables.b  = zero(T)
    variables.ξ .= ones(T,1,Data.N)
    variables.q .= ones(T,1,Data.N)
    variables.λ1 .= ones(T,1,Data.N)
    variables.λ2 .= ones(T,1,Data.N)
end

function variables_finalize!(
    variables::SvmVariables{T},
    equil::SvmEquilibration{T},
    status::SolverStatus
) where {T}

    return

end


function variables_rescale!(variables)

    return

end

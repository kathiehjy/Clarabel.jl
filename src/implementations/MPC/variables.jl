using LinearAlgebra

function variables_calc_mu(
    variables::MPCVariables{T},
) where {T}
  a = length(variables.λ)
  # λ and q are defined as very long vector with length h*N
  μ = (variables.λ ⋅ variables.q)/a
  return μ
end



function variables_calc_step_length(
    variables::MPCVariables{T},
    step::MPCVariables{T},
    cones::CompositeCone{T},
    settings::Settings{T},
    steptype::Symbol
) where {T}

    α = one(T)

    # λ, q are constrainted to be positive, 
    # so there is not need to check for sign, like for τ and κ
    (αλ,αq) = step_length(cones, step.λ, step.q, variables.λ, variables.q, settings, α)
    α = min(αλ, αq)

    if(steptype == :combined)
        α *= settings.max_step_fraction
    end

    return α   

end



function variables_barrier(
    variables::MPCVariables{T},
    step::MPCVariables{T},
    α::T,
    cones::CompositeCone{T},
) where {T}

    error("Function not yet implemented")

    # return barrier function
    # Leave empty
end



function variables_copy_from(dest::MPCVariables{T},src::MPCVariables{T}) where {T}

    dest.x .= src.x
    dest.x_end = src.x_end
    dest.u .= src.u
    dest.v .= src.v 
    dest.λ .= src.λ
    dest.q .= src.q

end



function variables_scale_cones!(
    variables::MPCVariables{T},
    cones::CompositeCone{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    error("Function not yet implemented")
end


function variables_add_step!(
    variables::MPCVariables{T},
    step::MPCVariables{T}, α::T
) where {T}

    # The first element of matrix x corresponds to x0, which is known
    # Only x₁~xₙ need to update
    @. variables.x[:,2:end] += α*step.x[:,2:end]
    @. variables.x_end      += α*step.x_end
    @. variables.u          += α*step.u
    @. variables.v          += α*step.v
    @. variables.λ          += α*step.λ 
    @. variables.q          += α*step.q 

    return nothing
end





function variables_affine_step_rhs!(
    d::MPCVariables{T},
    r::MPCResiduals{T},
    variables::MPCVariables{T},
) where{T}

    @. d.x     =  r.r1
    @. d.u     =  r.r2
    @. d.λ_m   =  r.r3
    @. d.v     =  r.r4
    d.λ       .=  variables.q .* variables.λ
    @. d.x_end =  r.r_end 

    return nothing
end


function variables_combined_step_rhs!(
    d::MPCVariables{T},
    r::MPCResiduals{T},
    variables::MPCVariables{T},
    step::MPCVariables{T},
    σ::T,
    μ::T
) where {T}


    dotσμ = σ * μ
    @. d.x     = (one(T) - dotσμ)*r.r1
    @. d.u     = (one(T) - dotσμ)*r.r2
    @. d.λ_m   = (one(T) - dotσμ)*r.r3
    @. d.v     = (one(T) - dotσμ)*r.r4
    d.λ       .= variables.q .*variables.λ + step.q .*step.λ .+dotσμ
    @. d.x_end = (one(T) - dotσμ)*r.r_end 

end

# Calls shift_to_cone on all conic variables and does not
# touch the primal variables. Used for symmetric problems.

function variables_symmetric_initialization!(
    variables::MPCVariables{T},
    cones::CompositeCone{T}
) where {T}

    """Leave empty
    """
    shift_to_cone!(cones,variables.q)
    shift_to_cone!(cones,variables.λ)


    variables.q .= 1
    variables.λ .= 1

end


# Calls unit initialization on all conic variables and zeros
# the primal variables.   Used for nonsymmetric problems.
function variables_unit_initialization!(
    variables::MPCVariables{T},
    data::MPCProblemData{T}
) where {T}

    variables.x     .= zero(T)
    variables.x[:,1].= data.x0
    variables.x_end .= zero(T)
    variables.u     .= zero(T)
    variables.v     .= zero(T)
    # λ and q are known to be NonnegativeCone,
    # so no need to call generic initilization method for conic objects
    variables.λ     .= one(T)
    variables.q     .= one(T)


end

function variables_finalize!(
    variables::MPCVariables{T},
    equil,
    status::SolverStatus
) where {T}

    return

end


function variables_rescale!(variables)

    return

end

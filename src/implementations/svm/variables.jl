
function variables_calc_mu(
    variables::SvmVariables{T},
    residuals::SvmResiduals{T},
    cones::CompositeCone{T}
) where {T}

  error("Function not yet implemented")

  return μ
end


function variables_calc_step_length(
    variables::SvmVariables{T},
    step::SvmVariables{T},
    cones::CompositeCone{T},
    settings::Settings{T},
    steptype::Symbol
) where {T}

    error("Function not yet implemented")

end


function variables_barrier(
    variables::SvmVariables{T},
    step::SvmVariables{T},
    α::T,
    cones::CompositeCone{T},
) where {T}

    error("Function not yet implemented")

    #return barrier function
end

function variables_copy_from(dest::SvmVariables{T},src::SvmVariables{T}) where {T}
    error("Function not yet implemented")
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

    error("Function not yet implemented")

    return nothing
end


function variables_affine_step_rhs!(
    d::SvmVariables{T},
    r::SvmResiduals{T},
    variables::SvmVariables{T},
    cones::CompositeCone{T}
) where{T}

    error("Function not yet implemented")

    return nothing
end


function variables_combined_step_rhs!(
    d::SvmVariables{T},
    r::SvmResiduals{T},
    variables::SvmVariables{T},
    cones::CompositeCone{T},
    step::SvmVariables{T},
    σ::T,
    μ::T
) where {T}

    error("Function not yet implemented")

end

# Calls shift_to_cone on all conic variables and does not
# touch the primal variables. Used for symmetric problems.

function variables_symmetric_initialization!(
    variables::SvmVariables{T},
    cones::CompositeCone{T}
) where {T}

    error("Function not yet implemented")
end


# Calls unit initialization on all conic variables and zeros
# the primal variables.   Used for nonsymmetric problems.
function variables_unit_initialization!(
    variables::SvmVariables{T},
    cones::CompositeCone{T}
) where {T}

    error("Function not yet implemented")
end

function variables_finalize!(
    variables::SvmVariables{T},
    equil::SvmEquilibration{T},
    status::SolverStatus
) where {T}

    error("Function not yet implemented")

end


function variables_rescale!(variables)

    error("Function not yet implemented")

end

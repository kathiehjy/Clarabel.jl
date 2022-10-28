
function variables_calc_mu(
    variables::TemplateVariables{T},
    residuals::TemplateResiduals{T},
    cones::CompositeCone{T}
) where {T}

  error("Function not yet implemented")

  return μ
end


function variables_calc_step_length(
    variables::TemplateVariables{T},
    step::TemplateVariables{T},
    cones::CompositeCone{T},
    settings::Settings{T},
    steptype::Symbol
) where {T}

    error("Function not yet implemented")

end


function variables_barrier(
    variables::TemplateVariables{T},
    step::TemplateVariables{T},
    α::T,
    cones::CompositeCone{T},
) where {T}

    error("Function not yet implemented")

    #return barrier function
end

function variables_copy_from(dest::TemplateVariables{T},src::TemplateVariables{T}) where {T}
    error("Function not yet implemented")
end

function variables_scale_cones!(
    variables::TemplateVariables{T},
    cones::CompositeCone{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

    error("Function not yet implemented")
end


function variables_add_step!(
    variables::TemplateVariables{T},
    step::TemplateVariables{T}, α::T
) where {T}

    error("Function not yet implemented")

    return nothing
end


function variables_affine_step_rhs!(
    d::TemplateVariables{T},
    r::TemplateResiduals{T},
    variables::TemplateVariables{T},
    cones::CompositeCone{T}
) where{T}

    error("Function not yet implemented")

    return nothing
end


function variables_combined_step_rhs!(
    d::TemplateVariables{T},
    r::TemplateResiduals{T},
    variables::TemplateVariables{T},
    cones::CompositeCone{T},
    step::TemplateVariables{T},
    σ::T,
    μ::T
) where {T}

    error("Function not yet implemented")

end

# Calls shift_to_cone on all conic variables and does not
# touch the primal variables. Used for symmetric problems.

function variables_symmetric_initialization!(
    variables::TemplateVariables{T},
    cones::CompositeCone{T}
) where {T}

    error("Function not yet implemented")
end


# Calls unit initialization on all conic variables and zeros
# the primal variables.   Used for nonsymmetric problems.
function variables_unit_initialization!(
    variables::TemplateVariables{T},
    cones::CompositeCone{T}
) where {T}

    error("Function not yet implemented")
end

function variables_finalize!(
    variables::TemplateVariables{T},
    equil::TemplateEquilibration{T},
    status::SolverStatus
) where {T}

    error("Function not yet implemented")

end


function variables_rescale!(variables)

    error("Function not yet implemented")

end

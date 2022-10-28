function kkt_update!(
    kktsystem::TemplateKKTSystem{T},
    data::TemplateProblemData{T},
    cones::CompositeCone{T}
) where {T}

    error("Function not yet implemented")

    return is_success
end


function kkt_solve_initial_point!(
    kktsystem::TemplateKKTSystem{T},
    variables::TemplateVariables{T},
    data::TemplateProblemData{T}
) where{T}

    error("Function not yet implemented")

    return is_success

end


function kkt_solve!(
    kktsystem::TemplateKKTSystem{T},
    lhs::TemplateVariables{T},
    rhs::TemplateVariables{T},
    data::TemplateProblemData{T},
    variables::TemplateVariables{T},
    cones::CompositeCone{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    error("Function not yet implemented")

end

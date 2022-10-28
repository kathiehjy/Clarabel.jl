function kkt_update!(
    kktsystem::SvmKKTSystem{T},
    data::SvmProblemData{T},
    cones::CompositeCone{T}
) where {T}

    error("Function not yet implemented")

    return is_success
end


function kkt_solve_initial_point!(
    kktsystem::SvmKKTSystem{T},
    variables::SvmVariables{T},
    data::SvmProblemData{T}
) where{T}

    error("Function not yet implemented")

    return is_success

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

    error("Function not yet implemented")

end

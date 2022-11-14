using LinearAlgebra
function residuals_update!(
    residuals::SvmResiduals{T},     # residual at each iterations
    variables::SvmVariables{T},     # variable, i.e., current value of x, z
    data::SvmProblemData{T}
) where {T}

    residuals.rw .= variables.w - transpose(data.Y) * variables.λ2
    residuals.rξ .= variables.ξ + data.Y * variables.w - variable.b * data.y - variables.q - ones(T,data.N,1)
    residuals.rλ1.= -variables.λ1 - variables.λ2 + ones(T, 1, data.N) * data.C
    residuals.rλ2 = transpose(data.y) * variables.λ2

  return nothing
end

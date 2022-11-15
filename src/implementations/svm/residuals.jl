using LinearAlgebra
function residuals_update!(
    residuals::SvmResiduals{T},     # residual at each iterations
    variables::SvmVariables{T},     # variable, i.e., current value of x, z
    data::SvmProblemData{T}
) where {T}

    residuals.rw = variables.w - transpose(data.Y) * GetVector(variables.λ2)
    residuals.rξ = GetVector(variables.ξ) + data.Y * variables.w - variables.b * data.y - GetVector(variables.q) .- one(T)
    residuals.rλ1 = -GetVector(variables.λ1) - GetVector(variables.λ2) .+ one(T) * data.C
    residuals.rλ2 = data.y ⋅ variables.λ2

  return nothing
end

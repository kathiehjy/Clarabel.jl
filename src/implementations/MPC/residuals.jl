function residuals_update!(
    residuals::MPCResiduals{T},
    variables::MPCVariables{T},
    data::MPCProblemData{T}
) where {T}
  # If x, u and v are stored as matrix
  # λ, q are very long ConicVectors
  # This facilitate the computation of μ 

  
  # Construct matrix form of λ and q (Once implement successfully, preallocate memory for these matrix)
  h = data.h
  variables.λ_m = reshape(GetVector(variables.λ), h, :)
  variables.q_m = reshape(GetVector(variables.q), h, :)

  # Construct the matrix for xₖ₊₁
  x_one_step_ahead = Matrix(undef, data.n, data.N)
  x_one_step_ahead[:,1:N-1] = deepcopy(data.x[:,2:end])
  x_one_step_ahead[:,end] = deepcopy(data.x_end)

  # Compute residual matrix
  residuals.r1 = data.Q * variables.x - transpose(data.G) * λ_m + transpose(data.A) * variables.v 
  residuals.r2 = data.R * variables.u + transpose(data.D) * λ_m + transpose(data.B) * variables.v 
  residuals.r3 =-data.G * variables.x + data.D * variables.u + variables.q_m .- variables.d
  residuals.r4 = data.A * variables.x + data.B * variables.u - x_one_step_ahead
  residuals.r_end = data.Q̅ * variables.x_end + variables.v[:,end]

  return nothing
end

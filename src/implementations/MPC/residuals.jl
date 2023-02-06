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
  x_one_step_ahead[:,1:end-1] = deepcopy(variables.x[:,2:end])
  x_one_step_ahead[:,end] = deepcopy(variables.x_end)

  """computation of residual matrix r1 is different for different k values
  computation of residual matrix r2, r3, r4 are the same for different k values
  """
  # when k = 0, don't have r10 as x0 is known
  residuals.r1[:,1] = zeros(h)

  # when k = 1, 2, ... , N-1
  residuals.r1[:,2:end] = data.Q * variables.x[:,2:end] - transpose(data.G) * variables.λ_m[:,2:end] + transpose(data.A) * variables.v[:,2:end] - variables.v[:,1:end-1] 
  
  residuals.r2 = data.R * variables.u + transpose(data.D) * variables.λ_m + transpose(data.B) * variables.v 
  residuals.r3 =-data.G * variables.x + data.D * variables.u + variables.q_m .- data.d
  residuals.r4 = data.A * variables.x + data.B * variables.u - x_one_step_ahead

  # when k = N
  residuals.r_end = data.Q̅ * variables.x_end - variables.v[:,end]

  return nothing
end


function solution_finalize!(
	solution::MPCSolution{T},
	data::MPCProblemData{T},
	variables::MPCVariables{T},
	info::MPCInfo{T},
	settings::Settings{T}
) where {T}

	solution.status  = info.status
	solution.obj_val = info.cost_primal

	# Haven't include feasibility check for MPC problem, now is the same as SVM problem
	# Since SVM problem is always feasible, no need for feasibility certificate
	# Just copy from variable to the solution
	#copy internal variables and undo homogenization
	solution.x  .= variables.x
	solution.x_end  .= variables.x_end
	solution.u  .= variables.u
	solution.v  .= variables.v
	solution.λ  .= variables.λ 
	solution.q  .= variables.q


	solution.iterations  = info.iterations
	solution.solve_time  = info.solve_time
	solution.r_prim 	 = info.res_primal
	solution.r_dual      = info.res_dual

	return nothing


end

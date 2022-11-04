
function solution_finalize!(
	solution::SvmSolution{T},
	data::SvmProblemData{T},
	variables::SvmVariables{T},
	info::SvmInfo{T},
	settings::Settings{T}
) where {T}

	solution.status  = info.status
	solution.obj_val = info.cost_primal

	# Since SVM problem is always feasible, no need for feasibility certificate
	# Just copy from variable to the solution
	#copy internal variables and undo homogenization
	solution.w  .= variables.w
	solution.b   = variables.b
	solution.ξ  .= variables.ξ
	solution.λ1 .= variables.λ1
	solution.λ2 .= variables.λ2
	solution.q  .= variables.q


	solution.iterations  = info.iterations
	solution.solve_time  = info.solve_time
	solution.r_prim 	   = info.res_primal
	solution.r_dual 	   = info.res_dual

	return nothing


end

function info_update!(
    info::MPCInfo{T},
    data::MPCProblemData{T},
    variables::MPCVariables{T},
    residuals::MPCResiduals{T},
    settings::Settings{T},
    timers::TimerOutput
) where {T}

    # Update the values for attributes of the SvmInfo
    # μ, step_length, sigma, iterations are updated in info_save_scalars
    info.cost_primal = 0.5*tr(transpose(variables.x)*data.Q*variables.x) + 0.5*tr(transpose(variables.u)*data.R*variables.u) + 0.5*tr(transpose(variables.x_end)*data.Q̅*variables.x_end)
   
    info.cost_dual = (0.5*transpose(variables.x[:,1])*data.Q-transpose(variables.λ_m[:,1])*data.G+transpose(variables.v[:,1])*data.A)*variables.x[:,1] - 0.5*tr(transpose(variables.x[:,2:end])*data.Q*variables.x[:,2:end]) - 0.5*tr(transpose(variables.u)*data.R*variables.u) - 0.5*tr(transpose(variables.x_end)*data.Q̅*variables.x_end) - sum(transpose(variables.λ_m) * data.d)


    # r1, r2, r3, r4 are matrix, need to find the maximum element as primal and dual residual
    info.res_primal = max(maximum(residuals.r3),maximum(residuals.r4))
    info.res_dual = max(maximum(residuals.r1),maximum(residuals.r2),maximum(residuals.r_end))

    #absolute and relative gaps
    info.gap_abs    = abs(info.cost_primal - info.cost_dual)
    if(info.cost_primal > zero(T) && info.cost_dual < zero(T))
        info.gap_rel = floatmax(T)
    else
        info.gap_rel = info.gap_abs / max(one(T),min(abs(info.cost_primal),abs(info.cost_dual)))
    end

    #κ/τ. For SVM problem κ = 0, τ = 1
    info.ktratio = 0

    #solve time so far (includes setup!)
    info_get_solve_time!(info,timers)

end

function info_check_termination!(
    info::MPCInfo{T},
    residuals::MPCResiduals{T},
    settings::Settings{T},
    iter::Int
) where {T}

    info.status = UNSOLVED  #ensure default state to start

    # optimality or infeasibility
    #---------------------
    _check_convergence_full(info,residuals,settings)

    # poor progress
    #----------------------
    if info.status == UNSOLVED && iter > 0 &&
        ( info.res_dual > info.prev_res_dual ||
          info.res_primal > info.prev_res_primal
        )

        # Poor progress at high tolerance.
        if info.ktratio < 100*eps(T) &&
            ( info.prev_gap_abs < settings.tol_gap_abs ||
              info.prev_gap_rel < settings.tol_gap_rel
            )
            info.status = INSUFFICIENT_PROGRESS
        end

        # Going backwards. Stop immediately if residuals diverge out of feasibility tolerance.
        if ( info.res_dual > settings.tol_feas &&
             info.res_dual > 100*info.prev_res_dual
           ) ||
           ( info.res_primal > settings.tol_feas &&
             info.res_primal > 100*info.prev_res_primal
           )
             info.status = INSUFFICIENT_PROGRESS
        end
    end


    # time / iteration limits
    #----------------------
    if info.status == UNSOLVED
        if settings.max_iter  == info.iterations
            info.status = MAX_ITERATIONS

        elseif info.solve_time > settings.time_limit
            info.status = MAX_TIME
        end
    end

    # return TRUE if we settled on a final status
    return is_done = info.status != UNSOLVED
end


function info_save_prev_iterate(
    info::MPCInfo{T},
    variables::MPCVariables{T},
    prev_variables::MPCVariables{T}
) where {T}

    info.prev_cost_primal = info.cost_primal
    info.prev_cost_dual   = info.cost_dual
    info.prev_res_primal  = info.res_primal
    info.prev_res_dual    = info.res_dual
    info.prev_gap_abs     = info.gap_abs
    info.prev_gap_rel     = info.gap_rel

    variables_copy_from(prev_variables,variables);
end

function info_reset_to_prev_iterate(
    info::MPCInfo{T},
    variables::MPCVariables{T},
    prev_variables::MPCVariables{T}
) where {T}

    info.cost_primal = info.prev_cost_primal
    info.cost_dual   = info.prev_cost_dual
    info.res_primal  = info.prev_res_primal
    info.res_dual    = info.prev_res_dual
    info.gap_abs     = info.prev_gap_abs
    info.gap_rel     = info.prev_gap_rel

    variables_copy_from(variables,prev_variables);
end

function info_save_scalars(
    info::MPCInfo{T},
    μ::T,
    α::T,
    σ::T,
    iter::Int
) where {T}

    info.μ = μ
    info.step_length = α
    info.sigma = σ
    info.iterations = iter

    return nothing
end


function info_reset!(
    info::MPCInfo{T},
    timers::TimerOutput
) where {T}

    info.status     = UNSOLVED
    info.iterations = 0
    info.solve_time = 0

    #reset the solve! timer, but keep the setup!
    reset_timer!(timers["solve!"])

    return nothing
end


function info_get_solve_time!(
    info::MPCInfo{T},
    timers::TimerOutput
) where {T}
    #TimerOutputs reports in nanoseconds
    info.solve_time = TimerOutputs.tottime(timers)*1e-9
    return nothing
end


function info_finalize!(
    info::MPCInfo{T},
    residuals::MPCResiduals{T},
    settings::Settings{T},
    timers::TimerOutput
) where {T}

    # if there was an error or we ran out of time
    # or iterations, check for partial convergence

    if (status_is_errored(info.status) ||
        info.status == MAX_ITERATIONS  ||
        info.status == MAX_TIME
    )
        _check_convergence_almost(info,residuals,settings)
    end

    # final check of timers
    info_get_solve_time!(info,timers)
    return nothing
end



# utility functions for convergence checking

function _check_convergence_full(info,residuals,settings)

    # "full" tolerances
    tol_gap_abs = settings.tol_gap_abs
    tol_gap_rel = settings.tol_gap_rel
    tol_feas    = settings.tol_feas
    tol_infeas_abs = settings.tol_infeas_abs
    tol_infeas_rel = settings.tol_infeas_rel
    tol_ktratio    = settings.tol_ktratio

    solved_status  = SOLVED
    pinf_status    = PRIMAL_INFEASIBLE
    dinf_status    = DUAL_INFEASIBLE

    _check_convergence(info,residuals,
                       tol_gap_abs,tol_gap_rel,tol_feas,
                       tol_infeas_abs,tol_infeas_rel,tol_ktratio,
                       solved_status,pinf_status,dinf_status)

end


function _check_convergence_almost(info,residuals,settings)

    # "almost" tolerances
    tol_gap_abs = settings.reduced_tol_gap_abs
    tol_gap_rel = settings.reduced_tol_gap_rel
    tol_feas    = settings.reduced_tol_feas
    tol_infeas_abs = settings.reduced_tol_infeas_abs
    tol_infeas_rel = settings.reduced_tol_infeas_rel
    tol_ktratio    = settings.reduced_tol_ktratio

    solved_status  = ALMOST_SOLVED
    pinf_status    = ALMOST_PRIMAL_INFEASIBLE
    dinf_status    = ALMOST_DUAL_INFEASIBLE

    _check_convergence(info,residuals,
                       tol_gap_abs,tol_gap_rel,tol_feas,
                       tol_infeas_abs,tol_infeas_rel,tol_ktratio,
                       solved_status,pinf_status,dinf_status)

end


function _check_convergence(
    info::MPCInfo{T},
    residuals::MPCResiduals{T},
    tol_gap_abs::T,
    tol_gap_rel::T,
    tol_feas::T,
    tol_infeas_abs::T,
    tol_infeas_rel::T,
    tol_ktratio::T,
    solved_status::SolverStatus,
    pinf_status::SolverStatus,
    dinf_status::SolverStatus,
) where {T}

    """Currently, info.ktratio is set to be 0
    """
    if info.ktratio < tol_ktratio && _is_solved(info, tol_gap_abs, tol_gap_rel, tol_feas)
        info.status = solved_status
    elseif info.ktratio > 1/tol_ktratio
        if _is_primal_infeasible(info, residuals, tol_infeas_abs, tol_infeas_rel)
            info.status = pinf_status
        elseif _is_dual_infeasible(info, residuals, tol_infeas_abs, tol_infeas_rel)
            info.status = dinf_status
        end
    end
end



function _is_solved(info, tol_gap_abs, tol_gap_rel, tol_feas)

    if( ((info.gap_abs < tol_gap_abs) || (info.gap_rel < tol_gap_rel))
        && (info.res_primal < tol_feas)
        && (info.res_dual   < tol_feas)
    )
        return true
    else
        return false
    end
end

function _is_primal_infeasible(info, residuals, tol_infeas_abs, tol_infeas_rel)

    if (residuals.dot_bz < -tol_infeas_abs) &&
        (info.res_primal_inf < -tol_infeas_rel * residuals.dot_bz)
        return true
    else
        return false
    end
end

function _is_dual_infeasible(info, residuals, tol_infeas_abs, tol_infeas_rel)

    if (residuals.dot_qx < -tol_infeas_abs) &&
            (info.res_dual_inf < -tol_infeas_rel * residuals.dot_qx)
        return true
    else
        return false
    end
end

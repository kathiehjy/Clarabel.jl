# -------------------------------------
# utility constructor that includes
# both object creation and setup
#--------------------------------------
function Solver(
    D::AbstractMatrix{T},
    C::T,
    kwargs...
) where{T <: AbstractFloat}

    s = Solver{T}()
    svm_setup!(s,D,C,kwargs...)
    return s
end

# -------------------------------------
# setup!
# -------------------------------------


"""
	setup!(solver, D, C, [settings])

Populates a [`Solver`](@ref) with a Dataset defined by `D`, and a weight coefficient `C` to weight between the conflicting objective of maximise the margin and identify data correctly,
cones will all be NonnegativeConeT for SvmProblem

The solver will be configured to solve the  SVM problem

All data matrices must be sparse.  

To solve the problem, you must make a subsequent call to [`solve!`](@ref)
"""
function svm_setup!(s,D,C,settings::Settings)
    #this allows total override of settings during setup
    s.settings = settings
    svm_setup!(s,D,C)
end

function svm_setup!(s,D,C; kwargs...)
    #this allows override of individual settings during setup
    settings_populate!(s.settings, Dict(kwargs))
    svm_setup!(s,D,C)
end

# main setup function
function svm_setup!(
    s::Solver{T},
    D::AbstractMatrix{T},
    C::T     # No need to pass in cone as for QP, the only cone that will be used here is the NonnegativeConeT, no dof
) where{T}

    #make this first to create the timers
    s.info    = SvmInfo{T}()

    @timeit s.timers "setup!" begin
        N = length(D)
        # Construct cones as composite cones
        s.cones  = CompositeCone{T}([NonnegativeCone{T}(N), NonnegativeCone{T}(N), NonnegativeCone{T}(N), NonnegativeCone{T}(N)])
        s.data   = SvmProblemData{T}(D,C)   # Need for dimension or sanity check for SvmProblem

        s.variables = SvmVariables{T}(s.data.n,s.data.N,s.cones)
        s.residuals = SvmResiduals{T}(s.data.n,s.data.N)

        #equilibrate problem data immediately on setup.
        #this prevents multiple equlibrations if solve!
        #is called more than once.

        @timeit s.timers "kkt init" begin
            s.kktsystem = SvmKKTSystem{T}(s.data,s.cones,s.settings)
        end

        # work variables for assembling step direction LHS/RHS
        s.step_rhs  = SvmVariables{T}(s.data.n,s.data.N,s.cones)
        s.step_lhs  = SvmVariables{T}(s.data.n,s.data.N,s.cones)

        # a saved copy of the previous iterate
        s.prev_vars = SvmVariables{T}(s.data.n,s.data.N,s.cones)

        # user facing results go here
        s.solution  = SvmSolution{T}(s.data.n,s.data.N)

    end

    return s
end


"""No need to check dimension for SVM problem"""
#=
# sanity check problem dimensions passed by user
"""Sanity check not need for SvmProblem"""
function _check_dimensions(P,q,A,b,cones)

    n = length(q)
    m = length(b)
    p = sum(cone -> nvars(cone), cones; init = 0)

    m == size(A)[1] || throw(DimensionMismatch("A and b incompatible dimensions."))
    p == m          || throw(DimensionMismatch("Constraint dimensions inconsistent with size of cones."))
    n == size(A)[2] || throw(DimensionMismatch("A and q incompatible dimensions."))
    n == size(P)[1] || throw(DimensionMismatch("P and q incompatible dimensions."))
    size(P)[1] == size(P)[2] || throw(DimensionMismatch("P not square."))

end
=#


# an enum for reporting strategy checkpointing
@enum StrategyCheckpoint begin 
    Update = 0   # Checkpoint is suggesting a new ScalingStrategy
    NoUpdate     # Checkpoint recommends no change to ScalingStrategy
    Fail         # Checkpoint found a problem but no more ScalingStrategies to try
end


# -------------------------------------
# solve!
# -------------------------------------

"""
	solve!(solver)

Computes the solution to the problem in a `Clarabel.Solver` previously defined in [`setup!`](@ref).
"""
function solve!(
    s::Solver{T}
) where{T}

    # initialization needed for first loop pass 
    iter   = 0
    σ = one(T) 
    α = zero(T)
    μ = typemax(T)

    # solver release info, solver config
    # problem dimensions, cone type etc
    @notimeit begin
        print_banner(s.settings.verbose)
        info_print_configuration(s.info,s.settings,s.data,s.cones)
        info_print_status_header(s.info,s.settings)
    end

    info_reset!(s.info,s.timers)


    @timeit s.timers "solve!" begin

        # initialize variables to some reasonable starting point
        @timeit s.timers "default start" solver_default_start!(s)

        @timeit s.timers "IP iteration" begin

        # ----------
        #  main loop
        # ----------

        scaling = PrimalDual::ScalingStrategy

        while true

            #update the residuals
            #--------------
            residuals_update!(s.residuals,s.variables,s.data)
            

            #calculate duality gap (scaled)
            #--------------
            μ = variables_calc_mu(s.variables)

            # record scalar values from most recent iteration.
            # This captures μ at iteration zero. 

            info_save_scalars(s.info, μ, α, σ, iter)

            #convergence check and printing
            #--------------

#=            info_update!(
                s.info,s.data,s.variables,
                s.residuals,s.settings,s.timers
            )
            @notimeit info_print_status(s.info,s.settings)
            isdone = info_check_termination!(s.info,s.residuals,s.settings,iter)

            # check for termination due to slow progress and update strategy
            if isdone
                (action,scaling) = _strategy_checkpoint_insufficient_progress(s,scaling) 
                if     action ∈ [NoUpdate,Fail]; break;
                elseif action === Update; continue; 
                end
            end # allows continuation if new strategy provided
=#

            #increment counter here because we only count
            #iterations that produce a KKT update 
            iter += 1
            println("ITER = ", iter)
            if(iter > 20)
                break
            end 

            """No scaling for SVM problem
            #update the scalings
            #--------------
            variables_scale_cones!(s.variables,s.cones,μ,scaling)
            """

            #Update the KKT system and the constant parts of its solution.
            #Keep track of the success of each step that calls KKT
            #--------------
            """Not used in here, didn't use the solver to solve for the system, only returns true"""
            @timeit s.timers "kkt update" begin
            is_kkt_solve_success = kkt_update!(s.kktsystem,s.data,s.cones)
            end


            #calculate the affine step
            #--------------
            """Not sure how each term in the rhs of the system is mapped into the a variable object, 
            it's not used by the kkt_solver currently"""
            variables_affine_step_rhs!(
                s.step_rhs, s.residuals,
                s.variables
            )


            """kkt_solver.jl directly solve the reduced system,
            is it still needed to check for existence"""
            @timeit s.timers "kkt solve" begin
            is_kkt_solve_success = is_kkt_solve_success && 
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, :affine
                )  # Solve for and Update s.step_lhs, which is the affine step size
            end

            # combined step only on affine step success 
            if is_kkt_solve_success

                #calculate step length and centering parameter
                #--------------
                α = solver_get_step_length(s,:affine,scaling)
                σ = _calc_centering_parameter(α)
  
                #calculate the combined step and length
                #--------------
                variables_combined_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, 
                    s.step_lhs, σ, μ
                )

                @timeit s.timers "kkt solve" begin
                is_kkt_solve_success =
                    kkt_solve!(
                        s.kktsystem, s.step_lhs, s.step_rhs,
                        s.data, s.variables, :combined
                    )
                end

            end

            # check for numerical failure and update strategy
            (action,scaling) = _strategy_checkpoint_numerical_error(s, is_kkt_solve_success, scaling) 
            if     action === NoUpdate; ();  #just keep going 
            elseif action === Update; α = zero(T); continue; 
            elseif action === Fail;   α = zero(T); break; 
            end
    

            #compute final step length and update the current iterate
            #--------------
            α = solver_get_step_length(s,:combined,scaling)

            # check for undersized step and update strategy
            (action,scaling) = _strategy_checkpoint_small_step(s, α, scaling)
            if     action === NoUpdate; ();  #just keep going 
            elseif action === Update; α = zero(T); continue; 
            elseif action === Fail;   α = zero(T); break; 
            end 

            # Copy previous iterate in case the next one is a dud
            info_save_prev_iterate(s.info,s.variables,s.prev_vars)

            variables_add_step!(s.variables,s.step_lhs,α)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

    end #end solve! timer

    # Check we if actually took a final step.  If not, we need 
    # to recapture the scalars and print one last line 
    if(α == zero(T))
        info_save_scalars(s.info, μ, α, σ, iter)
        #@notimeit info_print_status(s.info,s.settings)
    end 

    info_finalize!(s.info,s.residuals,s.settings,s.timers)  #halts timers
    solution_finalize!(s.solution,s.data,s.variables,s.info,s.settings)

    @notimeit info_print_footer(s.info,s.settings)

    return s.solution
end


function solver_default_start!(s::Solver{T}) where {T}

    # If there are only symmetric cones, use CVXOPT style initilization
    # Otherwise, initialize along central rays

    if (is_symmetric(s.cones))
        #set all scalings to identity (or zero for the zero cone)
        set_identity_scaling!(s.cones)
        #Refactor
        kkt_update!(s.kktsystem,s.data,s.cones)
        #solve for primal/dual initial points via KKT
        kkt_solve_initial_point!(s.variables)
        #fix up (z,s) so that they are in the cone
        variables_symmetric_initialization!(s.variables, s.cones)

    else
        #Assigns unit (z,s) and zeros the primal variables 
        variables_unit_initialization!(s.variables)
    end

    return nothing
end


function solver_get_step_length(s::Solver{T},steptype::Symbol,scaling::ScalingStrategy) where{T}

    # step length to stay within the cones
    α = variables_calc_step_length(
        s.variables, s.step_lhs,
        s.cones, s.settings, steptype
    )

    # additional barrier function limits for asymmetric cones
    """Since cones in SVM problem are NonnegativeCone, which is symmetric,
    so the following won't get activated"""
    if (!is_symmetric(s.cones) && steptype == :combined && scaling == Dual)
        αinit = α
        α = solver_backtrack_step_to_barrier(s,αinit)
    end
    return α
end


# check the distance to the boundary for asymmetric cones
function solver_backtrack_step_to_barrier(
    s::Solver{T}, αinit::T
) where {T}

    backtrack = s.settings.linesearch_backtrack_step
    α = αinit

    for j = 1:50
        barrier = variables_barrier(s.variables,s.step_lhs,α,s.cones)
        if barrier < one(T)
            return α
        else
            α = backtrack*α   #backtrack line search
        end
    end

    return α
end


# Mehrotra heuristic
function _calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end



function _strategy_checkpoint_insufficient_progress(s::Solver{T},scaling::ScalingStrategy) where {T} 

    if s.info.status != INSUFFICIENT_PROGRESS
        # there is no problem, so nothing to do
        return (NoUpdate::StrategyCheckpoint, scaling)
    else 
        #recover old iterate since "insufficient progress" often 
        #involves actual degradation of results 
        info_reset_to_prev_iterate(s.info,s.variables,s.prev_vars)

        # If problem is asymmetric, we can try to continue with the dual-only strategy
        if !is_symmetric(s.cones) && (scaling == PrimalDual::ScalingStrategy)
            s.info.status = UNSOLVED
            return (Update::StrategyCheckpoint, Dual::ScalingStrategy)
        else
            return (Fail::StrategyCheckpoint, scaling)
        end
    end

end 


function _strategy_checkpoint_numerical_error(s::Solver{T}, is_kkt_solve_success::Bool, scaling::ScalingStrategy) where {T}

    # if kkt was successful, then there is nothing to do 
    if is_kkt_solve_success
        return (NoUpdate::StrategyCheckpoint, scaling)
    end
    # If problem is asymmetric, we can try to continue with the dual-only strategy
    if !is_symmetric(s.cones) && (scaling == PrimalDual::ScalingStrategy)
        return (Update::StrategyCheckpoint, Dual::ScalingStrategy)
    else
        #out of tricks.  Bail out with an error
        s.info.status = NUMERICAL_ERROR
        return (Fail::StrategyCheckpoint,scaling)
    end
end 


function _strategy_checkpoint_small_step(s::Solver{T}, α::T, scaling::ScalingStrategy) where {T}

    if !is_symmetric(s.cones) &&
        scaling == PrimalDual::ScalingStrategy && α < s.settings.min_switch_step_length
        return (Update::StrategyCheckpoint, Dual::ScalingStrategy)

    elseif α <= min(zero(T), s.settings.min_terminate_step_length)
        s.info.status = INSUFFICIENT_PROGRESS
        return (Fail::StrategyCheckpoint,scaling)

    else
        return (NoUpdate::StrategyCheckpoint,scaling)
    end 
end 


# printing 

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end



# -------------------------------------
# getters accessing individual fields via function calls.
# this is necessary because it allows us to use multiple
# dispatch through these calls to access internal solver
# data within the MOI interface, which must also support
# the ClarabelRs wrappers
# -------------------------------------

get_solution(s::Solver{T}) where {T} = s.solution
get_info(s::Solver{T}) where {T} = s.info




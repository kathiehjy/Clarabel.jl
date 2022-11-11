using TimerOutputs

# -------------------------------------
# abstract type defs
# -------------------------------------
abstract type AbstractVariables{T <: AbstractFloat}   end
abstract type AbstractEquilibration{T <: AbstractFloat}   end
abstract type AbstractResiduals{T <: AbstractFloat}   end
abstract type AbstractProblemData{T <: AbstractFloat} end
abstract type AbstractKKTSystem{T <: AbstractFloat} end
abstract type AbstractKKTSolver{T <: AbstractFloat} end
abstract type AbstractInfo{T <: AbstractFloat} end
abstract type AbstractSolution{T <: AbstractFloat} end
abstract type AbstractSolver{T <: AbstractFloat}   end

# Scaling strategy for variables.  Defined
# here to avoid errors due to order of includes

@enum ScalingStrategy begin
    PrimalDual = 0
    Dual       = 1
end

# -------------------------------------
# top level solver type
# -------------------------------------

"""
	Solver{T <: AbstractFloat}()
Initializes an empty Clarabel solver that can be filled with problem data using:

    setup!(solver, P, q, A, b, cones, [settings]).

"""
mutable struct Solver{T <: AbstractFloat} <: AbstractSolver{T}

    data::Union{AbstractProblemData{T},Nothing}
    variables::Union{AbstractVariables{T},Nothing}
    cones::Union{AbstractCone{T},Nothing}
    residuals::Union{AbstractResiduals{T},Nothing}
    kktsystem::Union{AbstractKKTSystem{T},Nothing}
    info::Union{AbstractInfo{T},Nothing}
    step_lhs::Union{AbstractVariables{T},Nothing}
    step_rhs::Union{AbstractVariables{T},Nothing}
    prev_vars::Union{AbstractVariables{T},Nothing}
    solution::Union{AbstractSolution{T},Nothing}
    settings::Settings{T}
    timers::TimerOutput

end

#initializes all fields except settings to nothing
function Solver{T}(settings::Settings{T}) where {T}

    to = TimerOutput()
    #setup the main timer sections here and
    #zero them.   This ensures that the sections
    #exists if we try to clear them later
    @timeit to "setup!" begin (nothing) end
    @timeit to "solve!" begin (nothing) end
    reset_timer!(to["setup!"])
    reset_timer!(to["solve!"])

    Solver{T}(ntuple(x->nothing, fieldcount(Solver)-2)...,settings,to)
end

function Solver{T}() where {T}
    #default settings
    Solver{T}(Settings{T}())
end

#partial user defined settings
function Solver(d::Dict) where {T}
    Solver{T}(Settings(d))
end

Solver(args...; kwargs...) = Solver{DefaultFloat}(args...; kwargs...)

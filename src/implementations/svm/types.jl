# -------------------------------------
# Svm solver subcomponent implementations
# -------------------------------------


# ---------------
# problem data
# ---------------

mutable struct SvmProblemData{T} <: AbstractProblemData{T}
    D::AbstractMatrix{T}
    x::AbstractMatrix{T}  # Data features x
    y::Vector{T}          # Data labels y
    Y::AbstractMatrix{T}  # yx
    C::T                  # Weights between different objectives
    n::Integer            # number of features
    N::Integer            # number of data points

    function SvmProblemData{T}(
        D::AbstractMatrix{T},
        C::T,
    ) where {T}    # D in {x | y form}
        D = deepcopy(D)
        x = deepcopy(D[:,1:end-1])
        y = deepcopy(D[:,end])
        Y = zeros(size(x))
        for i in range(1,length(y))
            Y[i,:] = x[i,:] .* y[i]
        end
        C = C
        (N, n) = size(D)   # N - number of data points
        n = n - 1          # n - number of features
        new(x, y, Y, C, n, N)
    end

end

SvmProblemData(args...) = SvmProblemData{DefaultFloat}(args...)




# ---------------
# variables
# ---------------

mutable struct SvmVariables{T} <: AbstractVariables{T}

    w::Vector{T}
    b::T
    ξ::ConicVector{T}
    q::ConicVector{T}
    λ1::ConicVector{T}
    λ2::ConicVector{T}


    function SvmVariables{T}(
        n::Integer, # number of features     
        N::Integer  # number of data points
    ) where {T}   
        """ ξ,q,λ1 and λ2 are all NonnegativeCone with dimension of N 
        """
        w = Vector{T}(undef,n)
        b = T(1)
        ξ = NonnegativeConeT(N)
        q = NonnegativeConeT(N)
        λ1 = NonnegativeConeT(N)
        λ2 = NonnegativeConeT(N)
        new(w, b, ξ, q, λ1, λ2)
    end

end

SvmVariables(args...) = SvmVariables{DefaultFloat}(args...)



# ---------------
# residuals
# ---------------

mutable struct SvmResiduals{T} <: AbstractResiduals{T}

    rw::Vector{T}
    rξ::Vector{T}
    rλ1::Vector{T}
    rλ2::T

    function SvmResiduals{T}(
        n::Integer, 
        N::Integer
    ) where {T}

        rw = Vector{T}(undef,n)
        rξ = Vector{T}(undef,N)
        rλ1 = Vector{T}(undef,N)
        rλ2 = T(1)

        new(rw, rξ, rλ1, rλ2)
    end

end

SvmResiduals(args...) = SvmResiduals{DefaultFloat}(args...)



# ----------------------
# progress info
# ----------------------

mutable struct SvmInfo{T} <: AbstractInfo{T}
# Same as the default one, all these information is needed 
    μ::T
    sigma::T
    step_length::T
    iterations::UInt32
    cost_primal::T
    cost_dual::T
    res_primal::T
    res_dual::T
    res_primal_inf::T
    res_dual_inf::T
    gap_abs::T
    gap_rel::T
    ktratio::T

    # previous iterate
    prev_cost_primal::T
    prev_cost_dual::T
    prev_res_primal::T
    prev_res_dual::T
    prev_gap_abs::T
    prev_gap_rel::T

    solve_time::Float64
    status::SolverStatus

    function SvmInfo{T}() where {T}

        prevvals = ntuple(x->floatmax(T), 6);
        new((ntuple(x->0, fieldcount(DefaultInfo)-6-1)...,prevvals...,UNSOLVED)...)

    end

end

SvmInfo(args...) = SvmInfo{DefaultFloat}(args...)

# ---------------
# KKT System
# ---------------

mutable struct SvmKKTSystem{T} <: AbstractKKTSystem{T}
""" Encode the reduced KKT system
    Not sure
"""
    #the KKT system solver
    kktsolver::AbstractKKTSolver{T}

    #solution vector for reduced KKT system 
    w::Vector{T}
    b::T

    function SvmKKTSystem{T}(
        data::SvmProblemData{T},
        cones::CompositeCone{T},
        settings::Settings{T}
    ) where {T}

        #basic problem dimensions
        n = data.n

        #create the linear solver.  Always LDL for now
        kktsolver = DirectLDLKKTSolver{T}(data.P,data.A,cones,m,n,settings)

        #the LHS of the reduced solve
        w   = Vector{T}(undef,n)
        b   = T(1)


        return new(kktsolver,w,b)

    end

end

SvmKKTSystem(args...) = SvmKKTSystem{DefaultFloat}(args...)



# ---------------
# solver results
# ---------------

"""
    SvmSolution{T <: AbstractFloat}
Object returned by the Svm solver after calling `optimize!(model)`.

Fieldname | Description
---  | :--- | :---
w | Vector{T}| Primal variable
b | T | Primal variable
ξ | Vector{T}| Primal variable

λ1 | Vector{T}| Dual variable
λ2 | Vector{T}| Dual variable

q | Vector{T}| (Primal) set variable

status | Symbol | Solution status
obj_val | T | Objective value
solve_time | T | Solver run time
iterations | Int | Number of solver iterations
r_prim       | primal residual at termination
r_dual       | dual residual at termination
"""

mutable struct SvmSolution{T} <: AbstractSolution{T}

    w::Vector{T}
    b::T
    ξ::Vector{T}
    λ1::Vector{T}
    λ2::Vector{T}
    q::Vector{T}

    status::SolverStatus
    obj_val::T
    solve_time::T
    iterations::UInt32
    r_prim::T
    r_dual::T

end

function SvmSolution{T}(n,N) where {T <: AbstractFloat}
    w = Vector{T}(undef,n)
    b = T(1)
    ξ = Vector{T}(undef,N)
    λ1 = Vector{T}(undef,N)
    λ2 = Vector{T}(undef,N)
    q = Vector{T}(undef,N)

    # seemingly reasonable defaults
    status  = UNSOLVED
    obj_val = T(NaN)
    solve_time = zero(T)
    iterations = 0
    r_prim     = T(NaN)
    r_dual     = T(NaN)
    
    return SvmSolution{T}(w,b,ξ,λ1,λ2,q,status,obj_val,solve_time,iterations,r_prim,r_dual)
end

SvmSolution(args...) = SvmSolution{DefaultFloat}(args...)

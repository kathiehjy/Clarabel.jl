# -------------------------------------
# template solver subcomponent implementations
# -------------------------------------


# ---------------
# problem data
# ---------------

mutable struct MPCProblemData{T} <: AbstractProblemData{T}

    # Predefined quantities in MPC problem
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
    Q̅::AbstractMatrix{T}
    A::AbstractMatrix{T}
    B::AbstractMatrix{T}
    D::AbstractMatrix{T}
    G::AbstractMatrix{T}
    d::Vector{T}
    N::Integer
    n::Integer              # dim of states
    m::Integer              # dim of inputs
    h::Integer
    x0::Vector{T}

    function MPCProblemData{T}(
        Q::AbstractMatrix{T},
        R::AbstractMatrix{T},
        Q̅::AbstractMatrix{T},
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        D::AbstractMatrix{T},
        G::AbstractMatrix{T},
        d::Vector{T},
        N::Integer,
        x0::Vector{T}            # x0 is a known initial condition
    ) where {T}   
        Q = deepcopy(Q)
        R = deepcopy(R)
        Q̅ = deepcopy(Q̅)
        A = deepcopy(A)
        B = deepcopy(B)
        D = deepcopy(D)
        G = deepcopy(G)
        d = deepcopy(d)
        N = N
        (h, n) = size(G)
        m = size(R, 1)
        x0 = deepcopy(x0)

        new(Q, R, Q̅, A, B, D, G, d, N, n, m, h, x0)
    end


end

MPCProblemData(args...) = MPCProblemData{DefaultFloat}(args...)



# ---------------
# variables
# ---------------

mutable struct MPCVariables{T} <: AbstractVariables{T}

    x::Matrix{T}
    x_end::Vector{T}
    u::Matrix{T}
    v::Matrix{T}           # Lagranian multiplier for equility constraints
    λ::ConicVector{T}      # Lagranian multiplier for inequility constraints λ>=0
    q::ConicVector{T}      # Slack variable q>=0


    # The following quantities are only for computation and storage of residuals
    λ_m::Matrix{T}
    q_m::Matrix{T}
 


    function MPCVariables{T}(
        n::Integer, # dimension of states 
        m::Integer, # dimension of inputs   
        N::Integer, # number of steps
        h::Integer, # dimension of q and λ
        cones::CompositeCone{T}
    ) where {T}   
        """Each element of λ and q are all NonnegativeCone with dimension of N 
        """
        x = Matrix{T}(undef, n, N)
        x_end = Vector{T}(undef, n)
        u = Matrix{T}(undef, m, N)
        v = Matrix{T}(undef, n, N)

        cones = CompositeCone{T}([NonnegativeCone{T}(h*N)])
        λ = ConicVector{T}(cones)
        q = ConicVector{T}(cones)


        λ_m = Matrix{T}(undef, h, N)
        q_m = Matrix{T}(undef, h, N)

        new(x,x_end,u,v,λ,q,λ_m,q_m)
    end

end

MPCVariables(args...) = MPCVariables{DefaultFloat}(args...)



# ---------------
# residuals
# ---------------

mutable struct MPCResiduals{T} <: AbstractResiduals{T}

    r1::Matrix{T}
    r2::Matrix{T}
    r3::Matrix{T}
    r4::Matrix{T}
    r_end::Vector{T}


    function MPCResiduals{T}(
        h::Integer, 
        n::Integer,
        m::Integer,
        N::Integer,
    ) where {T}
        r1 = Matrix{T}(undef, n, N)
        r2 = Matrix{T}(undef, m, N)
        r3 = Matrix{T}(undef, h, N)
        r4 = Matrix{T}(undef, n, N)
        r_end = Vector{T}(undef, n)

        new(r1, r2, r3, r4, r_end)
    end


end

MPCResiduals(args...) = MPCResiduals{DefaultFloat}(args...)





# ----------------------
# progress info
# ----------------------

mutable struct MPCInfo{T} <: AbstractInfo{T}

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

    function MPCInfo{T}() where {T}

        prevvals = ntuple(x->floatmax(T), 6);
        new((ntuple(x->0, fieldcount(MPCInfo)-6-1)...,prevvals...,UNSOLVED)...)

    end

end

MPCInfo(args...) = MPCInfo{DefaultFloat}(args...)


# ---------------
# KKT System
# ---------------

mutable struct MPCKKTSystem{T} <: AbstractKKTSystem{T}

    # Problem specific fields go here.


        function MPCKKTSystem{T}(
            #constructor arguments go here
        ) where {T}

        #constructor details go here

    end

end

MPCKKTSystem(args...) = MPCKKTSystem{DefaultFloat}(args...)



# ---------------
# solver results
# ---------------

mutable struct MPCSolution{T} <: AbstractSolution{T}

    x::Matrix{T}
    x_end::Vector{T}
    u::Matrix{T}
    v::Matrix{T}           # Lagranian multiplier for equility constraints
    λ::Vector{T}      # Lagranian multiplier for inequility constraints λ>=0
    q::Vector{T} 


    status::SolverStatus
    obj_val::T
    solve_time::T
    iterations::UInt32
    r_prim::T
    r_dual::T

end

function MPCSolution{T}(n,m,h,N) where {T <: AbstractFloat}
    x = Matrix{T}(undef,n, N)
    x_end = Vector{T}(undef,n)
    u = Matrix{T}(undef,m, N)
    v = Matrix{T}(undef,n, N)
    λ = Vector{T}(undef,h*N)
    q = Vector{T}(undef,h*N)

    # seemingly reasonable defaults
    status  = UNSOLVED
    obj_val = T(NaN)
    solve_time = zero(T)
    iterations = 0
    r_prim     = T(NaN)
    r_dual     = T(NaN)
    
    return MPCSolution{T}(x,x_end,u,v,λ,q,status,obj_val,solve_time,iterations,r_prim,r_dual)
end



MPCSolution(args...) = MPCSolution{DefaultFloat}(args...)

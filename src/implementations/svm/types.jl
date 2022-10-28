# -------------------------------------
# Svm solver subcomponent implementations
# -------------------------------------

# ---------------
# variables
# ---------------

mutable struct SvmVariables{T} <: AbstractVariables{T}

    # Problem specific fields go here.

    function SvmVariables{T}(
        #constructor arguments go here
    ) where {T}

        #constructor details go here

    end

end

SvmVariables(args...) = SvmVariables{DefaultFloat}(args...)



# ---------------
# residuals
# ---------------

mutable struct SvmResiduals{T} <: AbstractResiduals{T}

    # Problem specific fields go here.

    function SvmResiduals{T}(
        #constructor arguments go here
        ) where {T}

        #constructor details go here
    end

end

SvmResiduals(args...) = SvmResiduals{DefaultFloat}(args...)


# ---------------
# problem data
# ---------------

mutable struct SvmProblemData{T} <: AbstractProblemData{T}

    # Problem specific fields go here.

    function SvmProblemData{T}(
        #constructor arguments go here
    ) where {T}

        #constructor details go here

    end

end

SvmProblemData(args...) = SvmProblemData{DefaultFloat}(args...)


# ----------------------
# progress info
# ----------------------

mutable struct SvmInfo{T} <: AbstractInfo{T}

    # Problem specific fields go here.

    function SvmInfo{T}(
        #constructor arguments go here
    ) where {T}

        #constructor details go here

    end

end

SvmInfo(args...) = SvmInfo{DefaultFloat}(args...)

# ---------------
# KKT System
# ---------------

mutable struct SvmKKTSystem{T} <: AbstractKKTSystem{T}

    # Problem specific fields go here.


        function SvmKKTSystem{T}(
            #constructor arguments go here
        ) where {T}

        #constructor details go here

    end

end

SvmKKTSystem(args...) = SvmKKTSystem{DefaultFloat}(args...)



# ---------------
# solver results
# ---------------

mutable struct SvmSolution{T} <: AbstractSolution{T}

    # Problem specific fields go here.

end

function SvmSolution{T}(m,n) where {T <: AbstractFloat}

    #constructor details go here
end

SvmSolution(args...) = SvmSolution{DefaultFloat}(args...)

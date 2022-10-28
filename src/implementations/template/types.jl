# -------------------------------------
# template solver subcomponent implementations
# -------------------------------------

# ---------------
# variables
# ---------------

mutable struct TemplateVariables{T} <: AbstractVariables{T}

    # Problem specific fields go here.

    function TemplateVariables{T}(
        #constructor arguments go here
    ) where {T}

        #constructor details go here

    end

end

TemplateVariables(args...) = TemplateVariables{DefaultFloat}(args...)



# ---------------
# residuals
# ---------------

mutable struct TemplateResiduals{T} <: AbstractResiduals{T}

    # Problem specific fields go here.

    function TemplateResiduals{T}(
        #constructor arguments go here
        ) where {T}

        #constructor details go here
    end

end

TemplateResiduals(args...) = TemplateResiduals{DefaultFloat}(args...)


# ---------------
# problem data
# ---------------

mutable struct TemplateProblemData{T} <: AbstractProblemData{T}

    # Problem specific fields go here.

    function TemplateProblemData{T}(
        #constructor arguments go here
    ) where {T}

        #constructor details go here

    end

end

TemplateProblemData(args...) = TemplateProblemData{DefaultFloat}(args...)


# ----------------------
# progress info
# ----------------------

mutable struct TemplateInfo{T} <: AbstractInfo{T}

    # Problem specific fields go here.

    function TemplateInfo{T}(
        #constructor arguments go here
    ) where {T}

        #constructor details go here

    end

end

TemplateInfo(args...) = TemplateInfo{DefaultFloat}(args...)

# ---------------
# KKT System
# ---------------

mutable struct TemplateKKTSystem{T} <: AbstractKKTSystem{T}

    # Problem specific fields go here.


        function TemplateKKTSystem{T}(
            #constructor arguments go here
        ) where {T}

        #constructor details go here

    end

end

TemplateKKTSystem(args...) = TemplateKKTSystem{DefaultFloat}(args...)



# ---------------
# solver results
# ---------------

mutable struct TemplateSolution{T} <: AbstractSolution{T}

    # Problem specific fields go here.

end

function TemplateSolution{T}(m,n) where {T <: AbstractFloat}

    #constructor details go here
end

TemplateSolution(args...) = TemplateSolution{DefaultFloat}(args...)

module Clarabel

    using SparseArrays, LinearAlgebra, Printf
    const DefaultFloat = Float64
    const DefaultInt   = LinearAlgebra.BlasInt
    const IdentityMatrix = UniformScaling{Bool}

    #version / release info
    include("./version.jl")

    #API for user cone specifications
    include("./cones/cone_api.jl")

    #cone type definitions
    include("./cones/cone_types.jl")
    include("./cones/compositecone_type.jl")

    #core solver components
    include("./settings.jl")
    include("./conicvector.jl")
    include("./statuscodes.jl")
    include("./types.jl")

    #direct LDL linear solve methods
    include("./kktsolvers/direct-ldl/includes.jl")

    #KKT solvers and solver level kktsystem
    include("./kktsolvers/kktsolver_defaults.jl")
    include("./kktsolvers/kktsolver_directldl.jl")

    """Choose between different implementations
    <add other problem type implementations here>
    """
    #default solver components implementation

    #include("./implementations/default/include.jl")

    #include("./implementations/svm/include.jl")

    include("./implementations/MPC/include.jl")

    # printing and top level solver
    """ Choose between different solver
    """
    #include("./info_print.jl")
    #include("./solver.jl")

    #include("./SVMinfo_print.jl")
    #include("./SVMsolver.jl")

    #include("./MPCinfo_print.jl")
    include("./MPCsolver.jl")

    #conic constraints.  Additional
    #cone implementations go here
    include("./cones/coneops_defaults.jl")
    include("./cones/coneops_zerocone.jl")
    include("./cones/coneops_nncone.jl")
    include("./cones/coneops_socone.jl")
    include("./cones/coneops_psdtrianglecone.jl")
    include("./cones/coneops_expcone.jl")
    include("./cones/coneops_powcone.jl")
    include("./cones/coneops_compositecone.jl")
    include("./cones/coneops_exppow_common.jl")
    include("./cones/coneops_symmetric_common.jl")

    #various algebraic utilities
    include("./utils/mathutils.jl")
    include("./utils/csc_assembly.jl")

    # #MathOptInterface for JuMP/Convex.jl
    # module MOImodule
    #      include("./MOI_wrapper/MOI_wrapper.jl")
    # end
    # const Optimizer{T} = Clarabel.MOImodule.Optimizer{T}

end





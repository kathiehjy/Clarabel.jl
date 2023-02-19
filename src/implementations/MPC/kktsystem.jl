function kkt_update!(
    kktsystem::MPCKKTSystem{T},
    data::MPCProblemData{T},
    cones::CompositeCone{T}
) where {T}

    #update the linear solver with new cones
    #kkt returns the ldl factorisation results and store the factor in is_success
    return true
    is_success  = kktsolver_update!(kktsystem.kktsolver,cones)  
    

    return is_success
end


function kkt_solve_initial_point!(
    kktsystem::MPCKKTSystem{T},
    variables::MPCVariables{T},
    data::MPCProblemData{T}
) where{T}

    error("Function not yet implemented")

    return is_success

end


function kkt_solve!(
    kktsystem::MPCKKTSystem{T},
    lhs::MPCVariables{T},
    rhs::MPCVariables{T},
    data::MPCProblemData{T},
    variables::MPCVariables{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    N = data.N   
    h = data.h
    n = data.n 
    m = data.m
    dim = 2 * n + h + m
    total_d = N * (dim)
    r1 = deepcopy(rhs.x)
    r2 = deepcopy(rhs.u)
    r3 = deepcopy(rhs.λ_m)
    r4 = deepcopy(rhs.v)
    Λ  = deepcopy(rhs.q_m)
    r_end = deepcopy(rhs.x_end)

    coefficient = zeros(total_d, total_d)
    RHS = Vector(undef, total_d)

    λ_m_copy = deepcopy(variables.λ_m) 
    q_m_copy = deepcopy(variables.q_m)

    K = Diagonal(inv.(λ_m_copy[:,1]))*Diagonal(q_m_copy[:,1])
    coefficient[1:m,1:m]             .= data.R
    coefficient[1:m,1+m:m+h]         .= transpose(data.D)
    coefficient[1:m,1+m+h:n+m+h]     .= transpose(data.B)
    coefficient[1+m:m+h,1:m]         .= data.D
    coefficient[1+m:m+h,1+m:m+h]     .= -K 
    coefficient[1+m+h:n+m+h,1:m]     .= data.B
    coefficient[1+m+h:n+m+h,dim-n+1:dim] .= -I(n)
    coefficient[dim-n+1:dim,m+h+1:m+h+n] .= -I(n) 


    RHS[1:m]         .= r2[:,1]
    RHS[m+1:m+h]     .= r3[:,1] - Diagonal(inv.(variables.λ_m[:,1])) * Λ[:,1]
    RHS[m+h+1:n+m+h] .= r4[:,1]

    for i in 1:N-1
        K = Diagonal(inv.(λ_m_copy[:,i+1]))*Diagonal(q_m_copy[:,i+1])
        # Construct the coefficient matrix for the MPC problem
        coefficient[i*dim+1-n:i*dim,i*dim+1-n:i*dim]                 .= data.Q
        coefficient[i*dim+1-n:i*dim,i*dim+1+m:i*dim+m+h]             .= -transpose(data.G)
        coefficient[i*dim+1-n:i*dim,i*dim+1+m+h:i*dim+n+m+h]         .= transpose(data.A)
        coefficient[i*dim+1:i*dim+m,i*dim+1:i*dim+m]                 .= data.R
        coefficient[i*dim+1:i*dim+m,i*dim+1+m:i*dim+m+h]             .= transpose(data.D)
        coefficient[i*dim+1:i*dim+m,i*dim+1+m+h:i*dim+n+m+h]         .= transpose(data.B)
        coefficient[i*dim+1+m:i*dim+m+h,i*dim+1-n:i*dim]             .= -data.G
        coefficient[i*dim+1+m:i*dim+m+h,i*dim+1:i*dim+m]             .= data.D
        coefficient[i*dim+1+m:i*dim+m+h,i*dim+1+m:i*dim+m+h]         .= -K
        coefficient[i*dim+1+m+h:i*dim+n+m+h,i*dim+1-n:i*dim]         .= data.A
        coefficient[i*dim+1+m+h:i*dim+n+m+h,i*dim+1:i*dim+m]         .= data.B
        coefficient[i*dim+1+m+h:i*dim+n+m+h,i*dim+1+m+h+n:i*dim+2*n+m+h] .= -I(n)
        coefficient[i*dim+1+m+h+n:i*dim+2*n+m+h,i*dim+1+m+h:i*dim+n+m+h] .= -I(n) 

        # Construct the RHS of the KKTSystem
        RHS[i*dim+1-n:i*dim]         .= r1[:,i+1]
        RHS[i*dim+1:i*dim+m]         .= r2[:,i+1]
        RHS[i*dim+m+1:i*dim+m+h]     .= r3[:,i+1] - Diagonal(inv.(variables.λ_m[:,i+1])) * Λ[:,i+1]
        RHS[i*dim+m+h+1:i*dim+n+m+h] .= r4[:,i+1]
    end

    coefficient[total_d-n+1:end,total_d-n+1:end]             .= data.Q̅
    RHS[total_d-n+1:end] = r_end

    result = -inv(coefficient) * RHS


    lhs.x[:,1] .= data.x0
    lhs.u[:,1] = result[1:m]
    lhs.λ_m[:,1] = result[1+m:m+h] 
    lhs.v[:,1] = result[1+m+h:n+m+h] 
    for i in 1:N-1
        lhs.x[:,i+1] = result[i*dim-n+1:i*dim] 
        lhs.u[:,i+1] = result[i*dim+1:i*dim+m]
        lhs.λ_m[:,i+1] = result[i*dim+1+m:i*dim+m+h] 
        lhs.v[:,i+1] = result[i*dim+1+m+h:i*dim+n+m+h] 
    end
    lhs.λ .= vec(lhs.λ_m)
    Ttotal = Diagonal(inv.(λ_m_copy)).*Diagonal(q_m_copy)
    lhs.q_m = -Ttotal*lhs.λ_m - Diagonal(inv.(variables.λ_m)) * Λ
    lhs.q .= vec(lhs.q_m)
    lhs.x_end = result[total_d-n+1:end]
    
    return true

end

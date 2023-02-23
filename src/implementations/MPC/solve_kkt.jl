# Solve the kkt system for the MPC
using LinearAlgebra

function solve_kkt(
    N,   
    r1,
    r2,
    r3,
    r4,
    r_end,
    Q,
    R,
    Q̅,
    A,
    B, 
    D,
    G,
    λ,
    q,           # all the listed residual and varible are known
)               # only need to solve the reduced system
    (h, n) = size(G)
    m = size(R, 1)
    dim = 2 * n + 2 * h + m
    total_d = N * (dim) + n


    # Construct the coefficient matrix for the MPC problem
    coefficient = zeros(total_d, total_d)
    for i in 0:N-1
        coefficient[i*dim+1:i*dim+n,i*dim+1:i*dim+n]                     .= Q
        coefficient[i*dim+1:i*dim+n,i*dim+1+n+m:i*dim+n+m+h]             .= -transpose(G)
        coefficient[i*dim+1:i*dim+n,i*dim+1+n+m+h:i*dim+2*n+m+h]         .= transpose(A)
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n:i*dim+n+m]             .= R
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n+m:i*dim+n+m+h]         .= transpose(D)
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n+m+h:i*dim+2*n+m+h]     .= transpose(B)
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1:i*dim+n]             .= -G
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1+n:i*dim+n+m]         .= D
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1+2*n+m+h:i*dim+dim]   .= I(h)
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+1:i*dim+n]         .= A
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+1+n:i*dim+n+m]     .= B
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+dim+1:i*dim+dim+n] .= -I(n)
        coefficient[i*dim+1+2*n+m+h:i*dim+dim,i*dim+1+n+m:i*dim+n+m+h]   .= Diagonal(q[:,i+1])
        coefficient[i*dim+1+2*n+m+h:i*dim+dim,i*dim+1+2*n+m+h:i*dim+dim] .= Diagonal(λ[:,i+1])
        coefficient[(i+1)*dim+1:(i+1)*dim+n,i*dim+1+n+m+h:i*dim+2*n+m+h] .= -I(n) 
    end
    #for i in 1:N-1
    #    coefficient[i*dim+1:i*dim+n, (i-1)*dim+1+n+m+h:(i-1)*dim+2*n+m+h].= -I(n) 
        
    #end
    #coefficient[total_d-n+1:end,total_d-h-2*n+1:total_d-h-n] .= -I(n)
    coefficient[total_d-n+1:end,total_d-n+1:end]             .= Q̅


    # Construct the RHS of the KKTSystem
    RHS = Vector(undef, total_d)
    for i in 0:N-1
        RHS[i*dim+1:i*dim+n]             .= r1[:,i+1]
        RHS[i*dim+n+1:i*dim+n+m]         .= r2[:,i+1]
        RHS[i*dim+n+m+1:i*dim+n+m+h]     .= r3[:,i+1]
        RHS[i*dim+n+m+h+1:i*dim+2*n+m+h] .= r4[:,i+1]
        RHS[i*dim+2*n+m+h+1:(i+1)*dim]   .= q[:,i+1] .* λ[:,i+1]
    end
    RHS[total_d-n+1:end] = r_end

    result = -inv(coefficient) * RHS

    Δx = Matrix(undef,n,N)
    Δu = Matrix(undef,m,N)
    Δv = Matrix(undef,n,N)
    Δλ = Matrix(undef,h,N)
    Δq = Matrix(undef,h,N)
    for i in 0:N-1
        Δx[:,i+1] = result[i*dim+1:i*dim+n]
        Δu[:,i+1] = result[i*dim+1+n:i*dim+n+m]
        Δλ[:,i+1] = result[i*dim+1+n+m:i*dim+n+m+h] 
        Δv[:,i+1] = result[i*dim+1+n+m+h:i*dim+2*n+m+h] 
        Δq[:,i+1] = result[i*dim+1+2*n+m+h:i*dim+2*n+m+2*h] 
    end
    Δx_end = result[total_d-n+1:end]

    return Δx, Δu, Δλ, Δv, Δq, Δx_end
end

N = 2
A = [1 5; 0 2]
B = [3; 1]
D = 1
G = Matrix([0 1])
λ = Matrix([0.5 0.5])
q = Matrix([1 1])
Q = [2 1; 0 2]
R = 0.5
Q̅ = [1 3; 5 3]
r1 = [1.5 0.1; 0.9 0.5]
r2 = Matrix([1 1])
r3 = Matrix([2 2])
r4 = [0.1 0.5; 0.9 1]
r_end = [2; 3]



(Δx1, Δu1, Δλ1, Δv1, Δq1, Δx_end1) = solve_kkt(N,r1,r2,r3,r4,r_end,Q,R,Q̅,A,B,D,G,λ,q)

(h, n) = size(G)
# Construct the matrix for xₖ₊₁
x_one_step_ahead = zeros(n, N)
x_one_step_ahead[:,1:N-1] = deepcopy(Δx1[:,2:end])
x_one_step_ahead[:,end] = deepcopy(Δx_end1)

λ_m = reshape(Δλ1, h, :)
q_m = reshape(Δq1, h, :)

r1[:,1] = Q * Δx1[:,1] - transpose(G) * λ_m[:,1] + transpose(A) * Δv1[:,1]
r1[:,2] = Q * Δx1[:,2] - transpose(G) * λ_m[:,2] + transpose(A) * Δv1[:,2] - Δv1[:,1]
r2 = R * Δu1 + transpose(D) * λ_m + transpose(B) * Δv1
r3 =-G * Δx1 + D * Δu1 + q_m
r4 = A * Δx1 + B * Δu1 - x_one_step_ahead
rλ = λ .* Δq1 + q .* Δλ1 + λ .* q 
r_end = Q̅ * Δx_end1 - Δv1[:,end]

println(r1)
println(r2)
println(r3)
println(r4)
println(rλ)
println(r_end)


x0 = Δx1[:,1]
# Solve for symmetric KKT system
function solve_symmetric_kkt(
    N,   
    r1,
    r2,
    r3,
    r4,
    r_end,
    Q,
    R,
    Q̅,
    A,
    B, 
    D,
    G,
    λ,
    q,           # all the listed residual and varible are known
)               # only need to solve the reduced system
    (h, n) = size(G)
    m = size(R, 1)
    dim = 2 * n + h + m
    total_d = N * (dim) + n


    coefficient = zeros(total_d, total_d)
    RHS = Vector(undef, total_d)
    for i in 0:N-1
        T = Diagonal(inv.(λ[:,i+1]))*Diagonal(q[:,i+1])
        # Construct the coefficient matrix for the MPC problem
        coefficient[i*dim+1:i*dim+n,i*dim+1:i*dim+n]                     .= Q
        coefficient[i*dim+1:i*dim+n,i*dim+1+n+m:i*dim+n+m+h]             .= -transpose(G)
        coefficient[i*dim+1:i*dim+n,i*dim+1+n+m+h:i*dim+2*n+m+h]         .= transpose(A)
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n:i*dim+n+m]             .= R
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n+m:i*dim+n+m+h]         .= transpose(D)
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n+m+h:i*dim+2*n+m+h]     .= transpose(B)
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1:i*dim+n]             .= -G
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1+n:i*dim+n+m]         .= D
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1+n+m:i*dim+n+m+h]     .= -T
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+1:i*dim+n]         .= A
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+1+n:i*dim+n+m]     .= B
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+dim+1:i*dim+dim+n] .= -I(n)
        coefficient[(i+1)*dim+1:(i+1)*dim+n,i*dim+1+n+m+h:i*dim+2*n+m+h] .= -I(n) 

        # Construct the RHS of the KKTSystem
        RHS[i*dim+1:i*dim+n]             .= r1[:,i+1]
        RHS[i*dim+n+1:i*dim+n+m]         .= r2[:,i+1]
        RHS[i*dim+n+m+1:i*dim+n+m+h]     .= r3[:,i+1] - T * λ[:,i+1]
        RHS[i*dim+n+m+h+1:i*dim+2*n+m+h] .= r4[:,i+1]
    end
    #for i in 1:N-1
    #    coefficient[i*dim+1:i*dim+n, (i-1)*dim+1+n+m+h:(i-1)*dim+2*n+m+h].= -I(n) 
    #end
    #coefficient[total_d-n+1:end,total_d-h-2*n+1:total_d-h-n] .= -I(n)
    coefficient[total_d-n+1:end,total_d-n+1:end]             .= Q̅
    RHS[total_d-n+1:end] = r_end

    result = -inv(coefficient) * RHS
    println("symmetric result:")
    println(result)

    Δx = Matrix(undef,n,N)
    Δu = Matrix(undef,m,N)
    Δv = Matrix(undef,n,N)
    Δλ = Matrix(undef,h,N)
    Δq = Matrix(undef,h,N)
    for i in 0:N-1
        Δx[:,i+1] = result[i*dim+1:i*dim+n]
        Δu[:,i+1] = result[i*dim+1+n:i*dim+n+m]
        Δλ[:,i+1] = result[i*dim+1+n+m:i*dim+n+m+h] 
        Δv[:,i+1] = result[i*dim+1+n+m+h:(i+1)*dim] 
    end
    Ttotal = Diagonal(inv.(λ)).*Diagonal(q)
    Δq = -Ttotal*(Δλ + λ)
    Δx_end = result[total_d-n+1:end]

    return Δx, Δu, Δλ, Δv, Δq, Δx_end
end


N = 2
A = [1 5; 0 2]
B = [3; 1]
D = 1
G = Matrix([0 1])
λ = Matrix([0.5 0.5])
q = Matrix([1 1])
Q = [2 1; 0 2]
R = 0.5
Q̅ = [1 3; 5 3]
r1 = [1.5 0.1; 0.9 0.5]
r2 = Matrix([1 1])
r3 = Matrix([2 2])
r4 = [0.1 0.5; 0.9 1]
r_end = [2; 3]
(Δx, Δu, Δλ, Δv, Δq, Δx_end) = solve_symmetric_kkt(N,r1,r2,r3,r4,r_end,Q,R,Q̅,A,B,D,G,λ,q)

#=
(h, n) = size(G)
# Construct the matrix for xₖ₊₁
x_one_step_ahead = zeros(n, N)
x_one_step_ahead[:,1:N-1] = deepcopy(Δx[:,2:end])
x_one_step_ahead[:,end] = deepcopy(Δx_end)

λ_m = reshape(Δλ, h, :)
q_m = reshape(Δq, h, :)

r1[:,1] = Q * Δx[:,1] - transpose(G) * λ_m[:,1] + transpose(A) * Δv[:,1]
r1[:,2] = Q * Δx[:,2] - transpose(G) * λ_m[:,2] + transpose(A) * Δv[:,2] - Δv[:,1]
r2 = R * Δu + transpose(D) * λ_m + transpose(B) * Δv 
r3 =-G * Δx + D * Δu + q_m
r4 = A * Δx + B * Δu - x_one_step_ahead
rλ = λ .* Δq + q .* Δλ + λ .* q 
r_end = Q̅ * Δx_end - Δv[:,end]

println(r1)
println(r2)
println(r3)
println(r4)
println(rλ)
println(r_end)

println(" ")
println(Δx1-Δx)
println(Δu1-Δu)
println(Δλ1-Δλ)
println(Δv1-Δv)
println(Δq1-Δq)
println(Δx_end1-Δx_end)

=#

# Solve for symmetric KKT system
function solve_symmetric_kkt_with_initial(
    N,
    x0,  # x0 is a known initial condition   
    r1,
    r2,
    r3,
    r4,
    r_end,
    Q,
    R,
    Q̅,
    A,
    B, 
    D,
    G,
    λ,
    q,           # all the listed residual and varible are known
)               # only need to solve the reduced system
    (h, n) = size(G)
    m = size(R, 1)
    dim = 2 * n + h + m
    total_d = N * (dim)


    coefficient = zeros(total_d, total_d)
    RHS = Vector(undef, total_d)

    K = Diagonal(inv.(λ[:,1]))*Diagonal(q[:,1])
    coefficient[1:m,1:m]             .= R
    coefficient[1:m,1+m:m+h]         .= transpose(D)
    coefficient[1:m,1+m+h:n+m+h]     .= transpose(B)
    coefficient[1+m:m+h,1:m]         .= D
    coefficient[1+m:m+h,1+m:m+h]     .= -K
    coefficient[1+m+h:n+m+h,1:m]     .= B
    coefficient[1+m+h:n+m+h,dim-n+1:dim] .= -I(n)
    coefficient[dim-n+1:dim,m+h+1:m+h+n] .= -I(n) 

  
    RHS[1:m]         .= r2[:,1]
    #RHS[m+1:m+h]     .= r3[:,1] - K * λ[:,1]
    RHS[m+1:m+h]     .= r3[:,1] - Diagonal(inv.(λ[:,1]))*(-10178.589294984804)
    RHS[m+h+1:n+m+h] .= r4[:,1]

    for i in 1:N-1
        K = Diagonal(inv.(λ[:,i+1]))*Diagonal(q[:,i+1])
        # Construct the coefficient matrix for the MPC problem
        coefficient[i*dim+1-n:i*dim,i*dim+1-n:i*dim]                 .= Q
        coefficient[i*dim+1-n:i*dim,i*dim+1+m:i*dim+m+h]             .= -transpose(G)
        coefficient[i*dim+1-n:i*dim,i*dim+1+m+h:i*dim+n+m+h]         .= transpose(A)
        coefficient[i*dim+1:i*dim+m,i*dim+1:i*dim+m]                 .= R
        coefficient[i*dim+1:i*dim+m,i*dim+1+m:i*dim+m+h]             .= transpose(D)
        coefficient[i*dim+1:i*dim+m,i*dim+1+m+h:i*dim+n+m+h]         .= transpose(B)
        coefficient[i*dim+1+m:i*dim+m+h,i*dim+1-n:i*dim]             .= -G
        coefficient[i*dim+1+m:i*dim+m+h,i*dim+1:i*dim+m]             .= D
        coefficient[i*dim+1+m:i*dim+m+h,i*dim+1+m:i*dim+m+h]         .= -K
        coefficient[i*dim+1+m+h:i*dim+n+m+h,i*dim+1-n:i*dim]         .= A
        coefficient[i*dim+1+m+h:i*dim+n+m+h,i*dim+1:i*dim+m]         .= B
        coefficient[i*dim+1+m+h:i*dim+n+m+h,i*dim+1+m+h+n:i*dim+2*n+m+h] .= -I(n)
        coefficient[i*dim+1+m+h+n:i*dim+2*n+m+h,i*dim+1+m+h:i*dim+n+m+h] .= -I(n) 
        Λ = Diagonal(q[:,i+1])*λ[:,i+1]
        # Construct the RHS of the KKTSystem
        RHS[i*dim+1-n:i*dim]         .= r1[:,i+1]
        RHS[i*dim+1:i*dim+m]         .= r2[:,i+1]
        #RHS[i*dim+m+1:i*dim+m+h]     .= r3[:,i+1] - Diagonal(inv.(λ[:,i+1]))*Λ
        RHS[i*dim+m+1:i*dim+m+h]     .= r3[:,i+1] - Diagonal(inv.(λ[:,i+1]))*(-4500.1892949848025)
        RHS[i*dim+m+h+1:i*dim+n+m+h] .= r4[:,i+1]
    end

    coefficient[total_d-n+1:end,total_d-n+1:end]             .= Q̅
    RHS[total_d-n+1:end] = r_end

    result = -inv(coefficient) * RHS

    println("initial result:")
    println(result)

    Δx = Matrix(undef,n,N)
    Δu = Matrix(undef,m,N)
    Δv = Matrix(undef,n,N)
    Δλ = Matrix(undef,h,N)
    Δq = Matrix(undef,h,N)
    Δx[:,1] = x0
    Δu[:,1] = result[1:m]
    Δλ[:,1] = result[1+m:m+h] 
    Δv[:,1] = result[1+m+h:n+m+h] 
    for i in 1:N-1
        Δx[:,i+1] = result[i*dim-n+1:i*dim] 
        Δu[:,i+1] = result[i*dim+1:i*dim+m]
        Δλ[:,i+1] = result[i*dim+1+m:i*dim+m+h] 
        Δv[:,i+1] = result[i*dim+1+m+h:i*dim+n+m+h] 
    end
    Ktotal = Diagonal(inv.(λ)).*Diagonal(q)
    #Δq = -Ktotal*(Δλ + λ)
    Δq = -Ktotal*Δλ - Diagonal(inv.(λ))*([-10178.589294984804 -4500.1892949848025])
    Δx_end = result[total_d-n+1:end]

    return Δx, Δu, Δλ, Δv, Δq, Δx_end
end



N = 2
#=A = [1 5; 0 2]
B = [3; 1]
D = 1
G = Matrix([0 1])
λ = Matrix([0.5 0.5])
q = Matrix([1 1])
Q = [2 1; 0 2]
R = 0.5
Q̅ = [1 3; 5 3]
r1 = [1.5 0.1; 0.9 0.5]
r2 = Matrix([1 1])
r3 = Matrix([2 2])
r4 = [0.1 0.5; 0.9 1]
r_end = [2; 3]=#

A = I(1)*1.
B = I(1)*1.
D = I(1)*1.
G = 1 *I(1)*1.
d = [100.]
N = 2
x0 = [1.]
R = I(1)*1.
Q = I(1)*1.
Q̅ = I(1)*1.
λ = Matrix([1. 1])
q = Matrix([1. 1])
r1 = Matrix([0. -1]).*0.0292949848018218
r2 = Matrix([1 1.]).*0.0292949848018218
r3 = Matrix([-100. -99]).*0.0292949848018218
r4 = Matrix([1 0.]).*0.0292949848018218
r_end = I(1)*0

(Δx, Δu, Δλ, Δv, Δq, Δx_end) = solve_symmetric_kkt_with_initial(N,x0,r1,r2,r3,r4,r_end,Q,R,Q̅,A,B,D,G,λ,q)


(h, n) = size(G)
# Construct the matrix for xₖ₊₁
x_one_step_ahead = zeros(n, N)
x_one_step_ahead[:,1:N-1] = deepcopy(Δx[:,2:end])
x_one_step_ahead[:,end] = deepcopy(Δx_end)

λ_m = reshape(Δλ, h, :)
q_m = reshape(Δq, h, :)

r1[:,1] = [0.] 
r1[:,2] = Q * Δx[:,2] - transpose(G) * λ_m[:,2] + transpose(A) * Δv[:,2] - Δv[:,1]
r2 = R * Δu + transpose(D) * λ_m + transpose(B) * Δv 
r31 = D * Δu + q_m  # gives the correct first column, coefficient changed due to the elimination of x0
r32 =-G * Δx + D * Δu + q_m  # gives the rest of columns, expression unaffected by x0
r41 = B * Δu - x_one_step_ahead  # gives the correct first column
r42 = A * Δx + B * Δu - x_one_step_ahead  # gives the rest of columns
rλ = λ .* Δq + q .* Δλ + λ .* q 
r_end = Q̅ * Δx_end - Δv[:,end]

println("new residual")
println(r1)
println(r2)
println("r3")
println(r31)
println(r32)
println("r4")
println(r41)
println(r42)
println(rλ)
println(r_end)


# The solution of solving the KKT system are not the same,
# as the simultaneous functions have changed 
# println(" ")
# println(Δx1-Δx)
# println(Δu1-Δu)
# println(Δλ1-Δλ)
# println(Δv1-Δv)
# println(Δq1-Δq)
# println(Δx_end1-Δx_end)

println(Δx)
println(Δu)
println(Δλ)
println(Δv)
println(Δq)
println(Δx_end)
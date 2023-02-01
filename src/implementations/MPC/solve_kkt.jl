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
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1:i*dim+n]             .= -G
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+1:i*dim+n]         .= A
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n:i*dim+n+m]             .= R
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1+n:i*dim+n+m]         .= D
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+1+n:i*dim+n+m]     .= B
        coefficient[i*dim+1:i*dim+n,i*dim+1+n+m:i*dim+n+m+h]             .= -transpose(G)
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n+m:i*dim+n+m+h]         .= transpose(D)
        coefficient[i*dim+1+2*n+m+h:i*dim+dim,i*dim+1+n+m:i*dim+n+m+h]   .= Diagonal(q[:,i+1])
        coefficient[i*dim+1:i*dim+n,i*dim+1+n+m+h:i*dim+2*n+m+h]         .= transpose(A)
        coefficient[i*dim+1+n:i*dim+n+m,i*dim+1+n+m+h:i*dim+2*n+m+h]     .= transpose(B)
        coefficient[i*dim+1+n+m:i*dim+n+m+h,i*dim+1+2*n+m+h:i*dim+dim]   .= I(h)
        coefficient[i*dim+1+2*n+m+h:i*dim+dim,i*dim+1+2*n+m+h:i*dim+dim] .= Diagonal(λ[:,i+1])
        coefficient[i*dim+1+n+m+h:i*dim+2*n+m+h,i*dim+dim+1:i*dim+dim+n] .= -I(n)

    end
    coefficient[total_d-n+1:end,total_d-h-2*n+1:total_d-h-n] .= -I(n)
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
r1 = [0.5 0.1; 0.5 0.5]
r2 = Matrix([1 1])
r3 = Matrix([2 2])
r4 = [0.1 0.5; 0.9 1]
r_end = [2; 3]
(Δx, Δu, Δλ, Δv, Δq, Δx_end) = solve_kkt(N,r1,r2,r3,r4,r_end,Q,R,Q̅,A,B,D,G,λ,q)

(h, n) = size(G)
# Construct the matrix for xₖ₊₁
x_one_step_ahead = zeros(n, N)
x_one_step_ahead[:,1:N-1] = deepcopy(Δx[:,2:end])
x_one_step_ahead[:,end] = deepcopy(Δx_end)

λ_m = reshape(Δλ, h, :)
q_m = reshape(Δq, h, :)

r1 = Q * Δx - transpose(G) * λ_m + transpose(A) * Δv 
r2 = R * Δu + transpose(D) * λ_m + transpose(B) * Δv 
r3 =-G * Δx + D * Δu + q_m
r4 = A * Δx + B * Δu - x_one_step_ahead
r_end = Q̅ * Δx_end + Δv[:,end]

println(r1)
println(r2)
println(r3)
println(r4)
println(r_end)
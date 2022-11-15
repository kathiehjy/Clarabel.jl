using LinearAlgebra
# Solving reduced linear system for Δw and Δb
function solving_reduced_kkt(
    D,   # Each row is [feature vector | label]
    rλ2,
    rλ1,
    rξ,
    rw,
    λ1,
    λ2,
    q,
    ξ         # all the listed residual and varible are known
)               # only need to solve the reduced system

    (N, n) = size(D)   # N - number of data points
    n = n - 1          # n - number of features

    # construct y and s
    s = zeros(N, 1)         # s is label vector
    x = zeros(N, n)         # feature matrix
    y = zeros(N, n)         # pairwise y = s * x
    for i in range(1, N)  # the ith iteration corresponds to ith data pair
        x[i,:] = D[i, 1:n]
        s[i] = D[i, end]
        @. y[i, :] = s[i] * x[i, :]
    end

    # Construct D, used repeatedly 
    Dξ = Diagonal(ξ)
    Dλ1 = Diagonal(λ1)
    Dλ2 = Diagonal(λ2)
    Dq = Diagonal(q)
    D = inv(inv(Dλ2)*Dq + inv(Dλ1)*Dξ)
    
    const1 = Dq * λ2
    const2 = Dξ * λ1
    # Construct the linear coefficient of linear system
    I1 = Matrix(1.0I, n, n)     # The identity matrix added to the left top corner
    TL = transpose(y) * D * y   # LT -- stand for top left element
    BL = -transpose(s) * D * y  # LB -- bottom left element 
    TR = transpose(BL)
    BR = transpose(s) * D * s
    #Linear_system_coef = [I1+TL, TR;BL, BR]
    top = hcat(I1+TL, TR)
    bottom = hcat(BL, BR)
    Linear_system_coef = vcat(top, bottom)
    

    # Construct the rhs of the linear system   rhs = [Trhs; Brhs]
    reuse = -rξ+inv(Dλ1)*const2+inv(Dλ1)*Dξ*rλ1-inv(Dλ2)*const1
    Trhs = -rw + transpose(y)*D*reuse
    Brhs = -rλ2 .- transpose(s)*D*reuse
    rhs = vcat(Trhs, Brhs)
    result = inv(Linear_system_coef) * rhs      # result = [Δw; Δb]
    Δw = result[1:end-1]
    Δb = result[end]

    # Solve for Δλ2, Δξ, Δq, Δλ1 given Δw, Δb
    Δλ2 = D * (-rξ -y*Δw + ξ + inv(Dλ1)*Dξ*rλ1 - q + s.*Δb)
    Δλ1 = rλ1 - Δλ2
    Δq = -q - inv(Dλ2)*Dq*Δλ2
    Δξ = -ξ - inv(Dλ1)*Dξ*Δλ1
    return Δw, Δb, Δλ1, Δλ2, Δξ, Δq, y, s
end



#Example
rλ1 = [2.0; 2; 3; 5; 3.0] * 0.05
rλ2 = 5.0
rξ = [1.0; 3; 8; 3; 5.0] * 0.03
rw = [3.0; 2] * 0.02
λ1 = [1.0; 5; 3; 4; 5.0] 
λ2 = [5.0; 1; 4; 4; 5.0] 
q = [1.0; 2; 9; 4; 5.0] 
ξ = [1.0; 2; 3; 4; 5.0] 
DataP = [1 2 1; 0 2 1; 1 1.5 1; 3 3 -1; 2 4 -1]

(Δw, Δb, Δλ1, Δλ2, Δξ, Δq, y, s) = solving_reduced_kkt(DataP,rλ2,rλ1,rξ,rw,λ1,λ2,q,ξ)

rwt = Δw - transpose(y) * Δλ2        
rξt = y*Δw + Δξ - Δq - s*Δb
rλ1t = -Δλ2 - Δλ1                    
rλ2t = transpose(s) * Δλ2            
Dξ = Diagonal(ξ)
Dλ1 = Diagonal(λ1)
Dλ2 = Diagonal(λ2)
Dq = Diagonal(q)


display(rλ1t)
display(rλ2t)
display(rξt)
display(rwt)
display(Dq*Δλ2 + Dλ2*Δq + Dq*λ2)
display(Dλ1*Δξ + Dξ*Δλ1 + Dλ1*ξ)

# Solve the original system
function full_linear(   D,   # Each row is [feature vector | label]
    rλ2,
    rλ1,
    rξ,
    rw,
    λ1,
    λ2,
    q,
    ξ         # all the listed residual and varible are known
    )
    (N, n) = size(D)   # N - number of data points
    n = n - 1          # n - number of features

    # construct y and s
    s = zeros(N, 1)         # s is label vector
    x = zeros(N, n)         # feature matrix
    y = zeros(N, n)         # pairwise y = s * x
    for i in range(1, N)    # the ith iteration corresponds to ith data pair
        x[i,:] = D[i, 1:n]
        s[i] = D[i, end]
        @. y[i, :] = s[i] * x[i, :]
    end

    # Construct D, used repeatedly 
    Dξ = Diagonal(ξ)
    Dλ1 = Diagonal(λ1)
    Dλ2 = Diagonal(λ2)
    Dq = Diagonal(q)

    # Construct coefficient matrix
    lhs = zeros(4*N+n+1, 4*N+n+1)
    # Filling first row
    lhs[1:n, 1:n] = Matrix(1.0I, n, n) 
    lhs[1:n, n+1:n+N] = -transpose(y)
    # Filling second row
    lhs[n+1:n+N, 1:n] = y
    lhs[n+1:n+N, n+N+1:n+2*N] = Matrix(1.0I, N, N)
    lhs[n+1:n+N, n+2*N+1:n+3*N] = -Matrix(1.0I, N, N)
    lhs[n+1:n+N, end] = -s
    # Filling third row
    lhs[n+N+1:n+2*N, n+1:n+N] = -Matrix(1.0I, N, N)
    lhs[n+N+1:n+2*N, n+3*N+1:n+4*N] = -Matrix(1.0I, N, N)
    # Filling forth row
    lhs[n+2*N+1, n+1:n+N] = transpose(s)
    # 5th
    lhs[n+2*N+2:n+3*N+1, n+1:n+N] = Dq
    lhs[n+2*N+2:n+3*N+1, n+1+2*N:n+3*N] = Dλ2
    # 6th
    lhs[n+3*N+2:end, n+N+1:n+2*N] = Dλ1
    lhs[n+3*N+2:end, n+3*N+1:n+4*N] = Dξ


    # Construct rhs
    rhs = zeros(4*N+n+1,1)
    rhs[1:n] = -rw
    rhs[n+1:n+N] = -rξ
    rhs[n+N+1:n+2*N] = -rλ1
    rhs[n+2*N+1] = -rλ2
    rhs[n+2*N+2:n+3*N+1] = -Dq * λ2
    rhs[n+3*N+2:end] = -Dλ1 * ξ
    result = inv(lhs)*rhs
    Δw = result[1:n]
    Δλ2 = result[n+1:n+N]
    Δξ = result[n+N+1:n+2*N]
    Δq = result[n+2*N+1:n+3*N]
    Δλ1 = result[n+3*N+1:n+4*N]
    Δb = result[end]
    return Δw, Δb, Δλ1, Δλ2, Δξ, Δq
end
(Δwt, Δbt, Δλ1t, Δλ2t, Δξt, Δqt) = full_linear(DataP,rλ2,rλ1,rξ,rw,λ1,λ2,q,ξ)
rwt = Δwt - transpose(y) * Δλ2t        
rξt = y*Δwt + Δξt - Δqt - s*Δbt
rλ1t = -Δλ2t - Δλ1t                    
rλ2t = transpose(s) * Δλ2t            
Dξ = Diagonal(ξ)
Dλ1 = Diagonal(λ1)
Dλ2 = Diagonal(λ2)
Dq = Diagonal(q)


#display(rλ1t)
#display(rλ2t)
#display(rξt)
#display(rwt)

#display(Dq*Δλ2t + Dλ2*Δqt + Dq*λ2)
#display(Dλ1*Δξt + Dξ*Δλ1t + Dλ1*ξ)

display(Δwt - Δw)
display(Δbt - Δb)
display(Δλ1t - Δλ1)
display(Δλ2t - Δλ2)
display(Δξt - Δξ)
display(Δqt - Δq)
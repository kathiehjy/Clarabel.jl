using LinearAlgebra
function MPCinterface(R, Q, Q̅, A, B, D, G, d, N, x0)
    """Extract P, q, A, b to use default solver to solve MPC problem
    x is a very long vector that stores x0, u0, x1 ... x(N-1), u(N-1), xN.
    Each xi ∈ ℝⁿ, ui ∈ ℝᵐ
    """
    (h,m) = size(D)
    n = size(Q,1)

    P = zeros((N+1)*n+N*m, (N+1)*n+N*m)
    for i in 0:N-1
        P[i*(m+n)+1:i*(m+n)+n, i*(m+n)+1:i*(m+n)+n] = Q
        P[i*(m+n)+1+n:i*(m+n)+n+m,i*(m+n)+1+n:i*(m+n)+n+m] = R
    end
    P[N*(m+n)+1:N*(n+m)+n,N*(m+n)+1:N*(n+m)+n] = Q̅

    q = zeros((N+1)*n+N*m)

    A1 = zeros((N+1)*n, (N+1)*n+N*m)
    A2 = zeros(N*h, (N+1)*n+N*m)
    A1[1:n,1:n] = I(n)
    for i in 0:N-1
        A1[(i+1)*n+1:(i+1)*n+n,i*(n+m)+1:i*(n+m)+n] = A
        A1[(i+1)*n+1:(i+1)*n+n,i*(n+m)+n+1:i*(n+m)+n+m] = B
        A1[(i+1)*n+1:(i+1)*n+n,i*(n+m)+n+m+1:i*(n+m)+n+m+n] = -I(n)
        A2[i*h+1:i*h+h, i*(n+m)+1:i*(n+m)+n] = -G
        A2[i*h+1:i*h+h, i*(n+m)+1+n:i*(n+m)+n+m] = D
    end
    A = [A1; A2]

    b1 = zeros((N+1)*n)
    b1[1:n] = x0
    b2 = zeros(N*h)
    for i in 0:N-1
        b2[i*h+1:i*h+h] = d
    end
    b = [b1; b2]
    return P, q, A, b, n, h, m
end
#=
Q = I(2)*1.
Q̅ = I(2)*3.
R = I(2)*2.
A = I(2)*0.5
B = I(2)*0.3
D = [1 0]
G = [0.3 0]
d = [1.]
N = 2
x0 = [1.; 1]
(P,A,b,n,h) = MPCinterface(R, Q, Q̅, A, B, D, G, d, N, x0)=#
using Clarabel, SparseArrays
include("./src/implementations/MPC/MPC_interface.jl")

A = I(1)*1.
B = I(1)*1.
D = I(1)*1.
G = 1 *I(1)*1.
d = [100.]
N = 1000
x0 = [1.]
R = I(1)*1.
Q = I(1)*1.
Q̅ = I(1)*5.

(P1,q1,A1,b,n,h,m) = MPCinterface(R, Q, Q̅, A, B, D, G, d, N, x0)
P = sparse(P1)*1.
q = sparse(q1)*1.
A = sparse(A1)*1.

cones =
    [Clarabel.ZeroConeT((N+1)*n),           #<--- for the equality constraint
     Clarabel.NonnegativeConeT(N*h)]    #<--- for the inequality constraints

settings = Clarabel.Settings()

solver   = Clarabel.Solver()

Clarabel.setup!(solver, P, q, A, b, cones, settings)

result = Clarabel.solve!(solver)

# result.x = [x0 u0 x1 u1 x2 u2 ... x(N-1) u(N-1) x(N)]
X = Matrix(undef,n,N)
U = Matrix(undef,m,N)

for i in 0:N-1
    X[:,i+1] = result.x[i*(n+m)+1:i*(n+m)+n]
    U[:,i+1] = result.x[i*(n+m)+1+n:i*(n+m)+n+m]
end
x_end = result.x[end-n+1:end] 
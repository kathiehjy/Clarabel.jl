using Clarabel, SparseArrays
D = [1. 2 -1; 0 2 -1; 1 1.5 -1; 3 3 1; 2 4 1; 1 8 1]   
C = 1.               
include("./src/implementations/svm/SVMinterface.jl")
(P_p, q_p, A_p, b_p, cones_p) = SVMinterface(D, C)
x = deepcopy(D[:,1:end-1])
y = deepcopy(D[:,end])
Y = zeros(size(x))
for i in range(1,length(y))
    Y[i,:] = x[i,:] .* y[i]
end
settings = Clarabel.Settings()

solver   = Clarabel.Solver()

Clarabel.setup!(solver, P_p, q_p, A_p, b_p, cones_p, settings)

result = Clarabel.solve!(solver)

# In the interface and the default problem correspond to hyperplane wᵀx + b = 0
# So Yw + yb - 1 + ξ - q = 0
# The OOQP paper and the linear system of the customized SVM solver correspond to hyperplane wᵀx - b = 0
q = Y * result.x[1:2] + y * result.x[3] + result.x[4:end] .- 1 


# Test for dual problem with variable = [λ1 λ2] 

# Construct P and q
P_Interest = Y * transpose(Y)
q_Interest = -ones(6)
P = sparse(zeros(12,12))
P[7:12,7:12] = P_Interest
q = zeros(12,1)
q[7:12] = q_Interest * 1.0      #To set up, have to use float number
# Construct A and b
# The first 6 rows of A correspond to λ1 + λ2 = C 
# The 7th row of A correspond to λ2ᵀy = 0
# The 8-13th rows correspond to λ1 >= 0
# The last 6 rows correspond to λ2 >= 0
A = sparse([1. 0. 0 0 0 0 1. 0 0 0 0 0;
    0 1. 0. 0 0 0 0 1 0 0 0 0;
    0 0 1 0 0 0 0. 0 1 0 0 0;
    0 0 0 1 0 0 0 0 0 1 0 0;
    0 0 0 0 1 0 0 0 0 0 1 0;
    0 0 0 0 0 1 0 0 0 0 0 1;
    0 0 0 0 0 0 0 0 0 0 0 0;
    -1 0 0 0 0 0 0 0 0 0 0 0;
    0 -1 0 0 0 0 0 0 0 0 0 0;
    0 0 -1 0 0 0 0 0 0 0 0 0;
    0 0 0 -1 0 0 0 0 0 0 0 0;
    0 0 0 0 -1 0 0 0 0 0 0 0;
    0 0 0 0 0 -1 0 0 0 0 0 0;
    0 0 0 0 0 0 -1 0 0 0 0 0;
    0 0 0 0 0 0 0 -1 0 0 0 0;
    0 0 0 0 0 0 0 0 -1 0 0 0;
    0 0 0 0 0 0 0 0 0 -1 0 0;
     0 0 0 0 0 0 0 0 0 0 -1 0;
     0 0 0 0 0 0 0 0 0 0 0 -1])
A[7, 7:12] = deepcopy(transpose(y))
b = zeros(19,1)
b[1:6] .= C * 1.0
cones =[Clarabel.ZeroConeT(7),           #<--- for the equality constraint
        Clarabel.NonnegativeConeT(12)]
"""Note: Currently all P, q, A, b have to be float number to avoid overflow during setup!
May need to change setup! to account for more flexible setup"""
settings = Clarabel.Settings()
solver   = Clarabel.Solver()
Clarabel.setup!(solver, P, q, A, b, cones, settings)
result = Clarabel.solve!(solver)
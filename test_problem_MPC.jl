using Clarabel
using LinearAlgebra
# Problem data for MPC problem

# # A, B, D, G, Q, R, Q̅ must be matrix
# A = [2. 1; 0.5 2]
# B = [1.; 3]
# B = reshape(B, length(B), 1)
# D = [1.; 1]
# D = reshape(D, length(D), 1)
# G = [1. 0; 0 1]
# N = 4

# # d and x0 need to be vector for setting up
# d = [10.; 10]
# x0 = [1.; 1]
# Q = [1 0.; 0 1]
# R = [0.01]
# R = reshape(R, length(R), 1)
# Q̅ = [1. 0; 0 1]
# settings = Clarabel.Settings(verbose = true)
# solver   = Clarabel.Solver()
# Clarabel.MPC_setup!(solver,Q,R,Q̅,A,B,D,G,d,N,x0,settings)
# result = Clarabel.solve!(solver)

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

settings = Clarabel.Settings(max_iter=20,verbose = true)
solver   = Clarabel.Solver()
Clarabel.MPC_setup!(solver,Q,R,Q̅,A,B,D,G,d,N,x0,settings)
result = Clarabel.solve!(solver)
using Clarabel
using LinearAlgebra
# Problem data for MPC problem
A = [2. 1; 0.5 2]
B = [1. 0]
D = [1. 1]
G = [1. 0; 0 1]
N = 4

# d and x0 need to be vector for setting up
d = [10.; 10]
x0 = [1.; 1]
Q = [1 0.; 0 1]
R = [0.01]
R = reshape(R, length(R), 1)
Q̅ = [1. 0; 0 1]
settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.MPC_setup!(solver,Q,R,Q̅,A,B,D,G,d,N,x0,settings)
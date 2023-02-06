using Clarabel

# Problem data for MPC problem
A = [2. 1; 0.5 2]
B = [1. 0]
D = [1. 1]
G = [1. 0; 0 1]
d = [10. 10]
N = 4
x0 = [1. 1]
Q = [1 0.; 0 1]
R = 0.01
Q̅ = [1. 0; 0 1]
settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.MPC_setup!(solver,Q,R,Q̅,A,B,D,G,d,N,x0,settings)
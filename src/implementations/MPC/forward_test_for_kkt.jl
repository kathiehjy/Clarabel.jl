""" Test for both the affine and combined step,
when using 1 instead of (1-σ*μ)
"""

# Calculate the residuals using the computed lhs step
step = solver.step_lhs
n = size(A,1)
# Check r2
println("r2 --- ")
println(R*step.u+D'*step.λ_m+B'*step.v) # = -r2
println(solver.residuals.r2)

# Check r3
println("r3 --- ")
println(-G*step.x+D*step.u+step.q_m)  # = -r3
println(solver.residuals.r3)

# Check r4
x_ahead = Matrix(undef, n, N)
x_ahead[:,1:end-1] = step.x[:,2:end]
x_ahead[:,end] = step.x_end[:]
println("r4 --- ")
println(A*step.x+B*step.u-x_ahead)   # = -r4
println(solver.residuals.r4)

# Check r1
v_pre = Matrix(undef,n,N-1)
v_pre[:,1:end] = step.v[:,1:end-1]
println("r1 --- ")
println(Q*step.x[:,2:end]-G'*step.λ_m[:,2:end]+A'*step.v[:,2:end]-v_pre)
println(solver.residuals.r1[:,2:end])

# Check r_end
println(Q̅*step.x_end - step.v[:,end])
println(solver.residuals.r_end)

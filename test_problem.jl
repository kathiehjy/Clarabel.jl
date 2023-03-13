using Clarabel
# Toy Example using SVMsolver 
# The data must be passed in as Float to execute the right setup! function

#D = [1. 1 1;-1 -1 -1]              # must pass in Float quantity  
#C = 1000.                          # must pass in Float quantity       
#Close to strict Constraint
# Expecting w = [-1, -1], b = 0
# get w = [-6.475, -6.475], b = 0 -- a diagonal line go through origin """

#D = [1. 2 1; 1 0 -1] 
#C = 100.
D = [1. 2 -1; 0 2 -1; 1 1.5 -1; 3 3 1; 2 4 1; 1 8 1]   
C = 1.               


settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.svm_setup!(solver, D, C, settings)
result = Clarabel.solve!(solver)  # Corresponds to hyperplane wᵀx - β = 0

using Clarabel
# Toy Example for svm problem
# The data must be passed in as Float to execute the right setup! function

D = [1. 1 1;-1 -1 -1]              # must pass in Float quantity  
C = 1000.                          # must pass in Float quantity       
#Close to strict Constraint
# Expecting w = [-1, -1], b = 0
# get w = [-6.475, -6.475], b = 0 -- a diagonal line go through origin """

                    


settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.svm_setup!(solver, D, C, settings)
result = Clarabel.solve!(solver)

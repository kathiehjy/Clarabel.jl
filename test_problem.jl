using Clarabel
# Toy Example for svm problem
# The data must be passed in as Float to execute the right setup! function

""" Example 1
D = [1. 1 1;-1 -1 -1]              # must pass in Float quantity  
C = 100.                            # must pass in Float quantity       
#Close to strict Constraint
# Expecting w = [-1, -1], b = 0
# get w = [-6.475, -6.475], b = 0 -- a diagonal line go through origin """

""" Example 2
D = [2. 0 -1;2 1 1] 
C = 1.  #give plane x2 = 0.5, w = [0, 2.]
"""
D = [1. 2 1; 0 2 1; 1 1.5 1; 3 3 -1; 2 4 -1; 1 8 -1]
C = 1.
             

"""Problem with current code is the singular problem when solving the linear system"""
settings = Clarabel.Settings(verbose = true, max_iter = 200)
solver   = Clarabel.Solver()
Clarabel.svm_setup!(solver, D, C, settings)
result = Clarabel.solve!(solver)   # Corresponds to hyperplane wᵀx - β = 0

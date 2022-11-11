# Toy Example for svm problem
D = [1. 1 1;-1 -1 -1]
C = 1000.   #Close to strict Constraint
"""Expecting w = -1, b = 0"""


settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.svm_setup!(solver, D, C, settings)
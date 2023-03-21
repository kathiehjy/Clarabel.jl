# A function aims to extract A, b, P, q from SVM problem formulation
# Date is stored in [x | y] forms, and stack all the data points as a matrix D
# C is the weighting between classification and maximum margin objectives
using Clarabel, SparseArrays, LinearAlgebra
function SVMinterface(D, C)   

    n = size(D, 2) - 1  # <-- n is the number of features
    l = size(D, 1)      # <-- l is the number of date points
    dim = n + l +1      # <-- A frequently appeared quantity

    # Construct P
    P1 = sparse(1.0I, n, n)
    P2 = spzeros(n, dim - n)
    P3 = spzeros(dim - n, dim - n)
    P = [P1 P2; transpose(P2) P3]  # Construct correctly -- tested

    # Construct q
    e1 = zeros(n + 1)
    e2 = ones(l)
    q = C*[e1; e2]                 # Construct correctly -- tested

    # Construct A
    A1 = zeros(l, dim)
    for i in range(1, l)  # the ith iteration corresponds to ith data pair
        xi = D[i, 1:n]
        yi = D[i,end]
        Yi = yi * xi
        A1[i, 1:n] = -Yi
        A1[i, n+1] = -yi
        A1[i, n+1+i] = -1
    end
    A2 = zeros(l, dim)
    for i in range(1, l)
        A2[i, n+1+i] = -1
    end
    A1  = sparse(A1);
    A2  = sparse(A2);
    A = [A1;A2]          # Construct correctly -- tested

    # Construct b
    b1 = -1* ones(l)
    b2 = zeros(l)
    b = [b1; b2]                  # Construct correctly -- tested

    # Construct cones
    cones = [Clarabel.NonnegativeConeT(2*l)]
    
    return P, q, A, b, cones  
    # Returns a tuple that contains all the data input required for clarabel solver
end 





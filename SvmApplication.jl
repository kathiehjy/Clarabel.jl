using CSV, LinearAlgebra
using DataFrames
using Clarabel 
# The data is stored as DataFrame 
df_train = CSV.read("/Users/huangjingyi/Desktop/4yp/MNIST dataset/mnist_train.csv", DataFrame)

# Convert the DataFrame object to Matrix
df_train1 = Matrix(df_train)
length_train = size(df_train1, 1)

"""Training using training dataset"""
# Consider the multi-class classification into binary classification -- “5” (1) or “not 5” (1)
# There are 5421 training example with label 5
for i in range(; length = length_train)
    if df_train1[i, 1] == 5
        df_train1[i, 1] = 1
    else
        df_train1[i, 1] = -1
    end
    i = i + 1
end


# Current use all the training data for training and no validation data
# In df_train1, the first element of each row is the label, the rest of the row are the features 
D_train = zeros(60000, size(df_train1,2))
D_train[:, 1:784] = deepcopy(df_train1[1:60000, 2:785]) * 1.
D_train[:, 785] = deepcopy(df_train1[1:60000, 1]) * 1.
#for i in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
C = 10.

settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.svm_setup!(solver, D_train, C, settings)
result = Clarabel.solve!(solver)  # Corresponds to hyperplane wᵀx - β = 0 

"""Obtain the parametric model
"""
# Corresponds to wᵀx - b = 0
w = result.w         
b = result.b


"""Test the parametric model and find the classification error for test dataset"""
# Load testing data
df_test = CSV.read("/Users/huangjingyi/Desktop/4yp/MNIST dataset/mnist_test.csv", DataFrame)

# Convert the DataFrame object to Matrix
df_test1 = Matrix(df_test)
length_test = size(df_test1, 1)

"""Testing using testing dataset"""
# Consider the multi-class classification into binary classification -- “5” (1) or “not 5” (1)
for i in range(; length = length_test)
    if df_test1[i, 1] == 5
        df_test1[i, 1] = 1
    else
        df_test1[i, 1] = -1
    end
    i = i + 1
end


# Extract the true labels of the test data
# In df_test1, the first element of each row is the label, the rest of the row are the features 
D_test = zeros(length_test, size(df_test1,2)-1)
D_true = deepcopy(df_test1[1:end, 1]) * 1.
D_test[:, 1:end] = deepcopy(df_test1[1:end, 2:end]) * 1.


# Store the predicted value
df_predicted = zeros(length_test,1)
for i in range(; length = length_test)
    prediction = D_test[i,:] ⋅ w - b
    if prediction > 0 
        df_predicted[i] = 1
    elseif prediction < 0
        df_predicted[i] = -1
    else
        df_predicted[i] = 0
    end
end

# Compute the classification error
# If the prediction is correct, it will give a 0 component in D_differ
D_differ = D_true - df_predicted    
correct = count(i->(i == 0), D_differ)
correct_rate = correct/length_test
println("No of examples classified correctly ",correct)
println("Percentage the classified correctly ",correct_rate)








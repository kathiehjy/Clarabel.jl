using CSV, LinearAlgebra
using DataFrames
using Clarabel 
"""Validate the optimal C value"""
# The data is stored as DataFrame 
df_train = CSV.read("/Users/huangjingyi/Desktop/4yp/MNIST dataset/mnist_train.csv", DataFrame)

# Convert the DataFrame object to Matrix
df_train1 = Matrix(df_train)
length_train = size(df_train1, 1)
feature_number = size(df_train1,2)

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

"""Choose between different C, using validation test
mnist_train is split into training data and validation data
"""
# Current use all the training data for training and no validation data
# In df_train1, the first element of each row is the label, the rest of the row are the features 
D_train1 = zeros(50000, feature_number)
D_train2 = zeros(50000, feature_number)
D_train3 = zeros(50000, feature_number)
D_train4 = zeros(50000, feature_number)
D_train5 = zeros(50000, feature_number)
D_train6 = zeros(50000, feature_number)

D_train1[:, 1:784] = deepcopy(df_train1[1:50000, 2:785]) * 1.
D_train1[:, 785] = deepcopy(df_train1[1:50000, 1]) * 1.
D_validate1 = deepcopy(df_train1[50001:end,2:785])
D_validate1_label = deepcopy(df_train1[50001:end,1])

D_train2[1:40000, 1:784] = deepcopy(df_train1[1:40000, 2:785]) * 1.
D_train2[1:40000, 785] = deepcopy(df_train1[1:40000, 1]) * 1.
D_train2[40001:end, 1:784] = deepcopy(df_train1[50001:end, 2:785]) * 1.
D_train2[40001:end, 785] = deepcopy(df_train1[50001:end, 1]) * 1.
D_validate2 = deepcopy(df_train1[40001:50000,2:785])
D_validate2_label = deepcopy(df_train1[40001:50000,1])

D_train3[1:30000, 1:784] = deepcopy(df_train1[1:30000, 2:785]) * 1.
D_train3[1:30000, 785] = deepcopy(df_train1[1:30000, 1]) * 1.
D_train3[30001:end, 1:784] = deepcopy(df_train1[40001:end, 2:785]) * 1.
D_train3[30001:end, 785] = deepcopy(df_train1[40001:end, 1]) * 1.
D_validate3 = deepcopy(df_train1[30001:40000,2:785])
D_validate3_label = deepcopy(df_train1[30001:40000,1])

D_train4[1:20000, 1:784] = deepcopy(df_train1[1:20000, 2:785]) * 1.
D_train4[1:20000, 785] = deepcopy(df_train1[1:20000, 1]) * 1.
D_train4[20001:end, 1:784] = deepcopy(df_train1[30001:end, 2:785]) * 1.
D_train4[20001:end, 785] = deepcopy(df_train1[30001:end, 1]) * 1.
D_validate4 = deepcopy(df_train1[20001:30000,2:785])
D_validate4_label = deepcopy(df_train1[20001:30000,1])

D_train5[1:10000, 1:784] = deepcopy(df_train1[1:10000, 2:785]) * 1.
D_train5[1:10000, 785] = deepcopy(df_train1[1:10000, 1]) * 1.
D_train5[10001:end, 1:784] = deepcopy(df_train1[20001:end, 2:785]) * 1.
D_train5[10001:end, 785] = deepcopy(df_train1[20001:end, 1]) * 1.
D_validate5 = deepcopy(df_train1[10001:20000,2:785])
D_validate5_label = deepcopy(df_train1[10001:20000,1])

D_train6[:, 1:784] = deepcopy(df_train1[10001:end, 2:785]) * 1.
D_train6[:, 785] = deepcopy(df_train1[10001:end, 1]) * 1.
D_validate6 = deepcopy(df_train1[1:10000,2:785])
D_validate6_label = deepcopy(df_train1[1:10000,1])

""" Finding the optimal C, using validation data
"""
#for i in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
C = 1.

settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
Clarabel.svm_setup!(solver, D_train1, C, settings)
result1 = Clarabel.solve!(solver)  # Corresponds to hyperplane wᵀx - β = 0 
Clarabel.svm_setup!(solver, D_train2, C, settings)
result2 = Clarabel.solve!(solver)  
Clarabel.svm_setup!(solver, D_train3, C, settings)
result3 = Clarabel.solve!(solver)  
Clarabel.svm_setup!(solver, D_train4, C, settings)
result4 = Clarabel.solve!(solver)  
Clarabel.svm_setup!(solver, D_train5, C, settings)
result5 = Clarabel.solve!(solver)  
Clarabel.svm_setup!(solver, D_train6, C, settings)
result6 = Clarabel.solve!(solver)  

"""Obtain the parametric model
"""
# Corresponds to wᵀx - b = 0
w1 = result1.w         
b1 = result1.b
w2 = result2.w         
b2 = result2.b
w3 = result3.w         
b3 = result3.b
w4 = result4.w         
b4 = result4.b
w5 = result5.w         
b5 = result5.b
w6 = result6.w         
b6 = result6.b

# Store the predicted value of the validation dataset
# 10000 datapoint in the validation set
df_predicted_v1 = zeros(10000,1)
df_predicted_v2 = zeros(10000,1)
df_predicted_v3 = zeros(10000,1)
df_predicted_v4 = zeros(10000,1)
df_predicted_v5 = zeros(10000,1)
df_predicted_v6 = zeros(10000,1)
for i in range(; length = 10000)
    prediction_v1 = D_validate1[i,:] ⋅ w1 - b1
    prediction_v2 = D_validate2[i,:] ⋅ w2 - b2
    prediction_v3 = D_validate3[i,:] ⋅ w3 - b3
    prediction_v4 = D_validate4[i,:] ⋅ w4 - b4
    prediction_v5 = D_validate5[i,:] ⋅ w5 - b5
    prediction_v6 = D_validate6[i,:] ⋅ w6 - b6

    if prediction_v1 > 0 
        df_predicted_v1[i] = 1
    elseif prediction_v1 < 0
        df_predicted_v1[i] = -1
    else
        df_predicted_v1[i] = 0
    end

    if prediction_v2 > 0 
        df_predicted_v2[i] = 1
    elseif prediction_v2 < 0
        df_predicted_v2[i] = -1
    else
        df_predicted_v2[i] = 0
    end

    if prediction_v3 > 0 
        df_predicted_v3[i] = 1
    elseif prediction_v3 < 0
        df_predicted_v3[i] = -1
    else
        df_predicted_v3[i] = 0
    end

    if prediction_v4 > 0 
        df_predicted_v4[i] = 1
    elseif prediction_v4 < 0
        df_predicted_v4[i] = -1
    else
        df_predicted_v4[i] = 0
    end

    if prediction_v5 > 0 
        df_predicted_v5[i] = 1
    elseif prediction_v5 < 0
        df_predicted_v5[i] = -1
    else
        df_predicted_v5[i] = 0
    end

    if prediction_v6 > 0 
        df_predicted_v6[i] = 1
    elseif prediction_v6 < 0
        df_predicted_v6[i] = -1
    else
        df_predicted_v6[i] = 0
    end
end


# Compute the classification error on the validation data
# If the prediction is correct, it will give a 0 component in D_differ
D_differ1 = D_validate1_label - df_predicted_v1   
D_differ2 = D_validate2_label - df_predicted_v2 
D_differ3 = D_validate3_label - df_predicted_v3 
D_differ4 = D_validate4_label - df_predicted_v4 
D_differ5 = D_validate5_label - df_predicted_v5 
D_differ6 = D_validate6_label - df_predicted_v6 

correct1 = count(i->(i == 0), D_differ1)
correct2 = count(i->(i == 0), D_differ2)
correct3 = count(i->(i == 0), D_differ3)
correct4 = count(i->(i == 0), D_differ4)
correct5 = count(i->(i == 0), D_differ5)
correct6 = count(i->(i == 0), D_differ6)
correct_rate1 = correct1/10000
correct_rate2 = correct2/10000
correct_rate3 = correct3/10000
correct_rate4 = correct4/10000
correct_rate5 = correct5/10000
correct_rate6 = correct6/10000

println("v1 ",correct1)
println("v1% ",correct_rate1)
println("v2 ",correct2)
println("v2% ",correct_rate2)
println("v3 ",correct3)
println("v3% ",correct_rate3)
println("v4 ",correct4)
println("v4% ",correct_rate4)
println("v5 ",correct5)
println("v5% ",correct_rate5)
println("v6 ",correct6)
println("v6% ",correct_rate6)


#=

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
D_true = deepcopy(df_test1[1:length_test, 1]) * 1.
D_test[:, 1:end] = deepcopy(df_test1[1:length_test, 2:end]) * 1.


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
=#

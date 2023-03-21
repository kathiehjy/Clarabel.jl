using CSV, LinearAlgebra, Statistics
using DataFrames
using Clarabel 

"""Validate the optimal C value, using abstraction"""
# validate function defined to do validation test using each (train, validate) set
function validate(
    D_train, C, features, true_label, solver, settings
)
    Clarabel.svm_setup!(solver, D_train, C, settings)
    result = Clarabel.solve!(solver)  # Corresponds to hyperplane wᵀx - β = 0 


    """Obtain the parametric model
    """
    # Corresponds to wᵀx - b = 0
    w = result.w         
    b = result.b

    # Store the predicted value of the validation dataset
    # 10000 datapoint in the validation set
    df_predicted_v = zeros(10000,1)

    for i in range(; length = 10000)
        prediction_v = features[i,:] ⋅ w - b

        if prediction_v > 0 
            df_predicted_v[i] = 1
        elseif prediction_v < 0
            df_predicted_v[i] = -1
        else
            df_predicted_v[i] = 0
        end

    end


    # Compute the classification error on the validation data
    # If the prediction is correct, it will give a 0 component in D_differ
    D_differ = true_label - df_predicted_v   
    correct = count(i->(i == 0), D_differ)
    correct_rate = correct/10000

  return correct, correct_rate
end


# The data is stored as DataFrame 
df_train = CSV.read("/Users/huangjingyi/Desktop/4yp/MNIST dataset/mnist_train.csv", DataFrame)

# Convert the DataFrame object to Matrix
df_train1 = Matrix(df_train)
length_train = size(df_train1, 1)
feature_number = size(df_train1,2)

# Consider the multi-class classification into binary classification -- “5” (1) or “not 5” (1)
# There are 5421 training example with label 5
for i in range(; length = length_train)
    if df_train1[i, 1] == 6
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
# Store the average correct rate for each C
Ave_rate = []
C_options = [0.0001, 0.001, 0.01]

settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()
for i in C_options
    C = i 

    (correct1, correct_rate1) = validate(D_train1,C,D_validate1,D_validate1_label,solver,settings)
    (correct2, correct_rate2) = validate(D_train2,C,D_validate2,D_validate2_label,solver,settings)
    (correct3, correct_rate3) = validate(D_train3,C,D_validate3,D_validate3_label,solver,settings)
    (correct4, correct_rate4) = validate(D_train4,C,D_validate4,D_validate4_label,solver,settings)
    (correct5, correct_rate5) = validate(D_train5,C,D_validate5,D_validate5_label,solver,settings)
    (correct6, correct_rate6) = validate(D_train6,C,D_validate6,D_validate6_label,solver,settings)

    ave_rate = mean([correct_rate1,correct_rate2,correct_rate3,correct_rate4,correct_rate5,correct_rate6])
    push!(Ave_rate,ave_rate)
    println("C ",C)
    println("ave% ",ave_rate)

end


"""Find the best C on average, train on the entire training set"""
(max_ave, index) = findmax(Ave_rate)
C_optimal = C_options[index]
D_train = zeros(60000, size(df_train1,2))
D_train[:, 1:784] = deepcopy(df_train1[1:60000, 2:785]) * 1.
D_train[:, 785] = deepcopy(df_train1[1:60000, 1]) * 1.
Clarabel.svm_setup!(solver, D_train, C_optimal, settings)
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
    if df_test1[i, 1] == 6
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

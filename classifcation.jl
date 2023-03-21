using CSV, LinearAlgebra, Statistics
using DataFrames
using Clarabel 

settings = Clarabel.Settings(verbose = true)
solver   = Clarabel.Solver()

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

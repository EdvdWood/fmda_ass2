#%% Importing

import assignment2_functions as fun
# Numpy is needed for calling arange to smoothe the curves.
import numpy as np

#%% Setting up Wrapper
def Wrapper(filename, country1, country2, indicator, tMax, folds, write_file):
    """
        Wrapper Function
        
        This function completes all the requirements set by Assignment 2.
        For individual components see the strings of these functions.
        
        Args:
            filename (string): The name or location of the file required.
            country1 (string): The name of the first country input.
            country2 (string): The name of the second country input.
            indicator (string): The indicator that has to be analyzed.
            tMax (integer): The final year for which to model towards. 
            folds (integer): The number of folds to use for the k-fold validation.
            write_file (string): The file which to write the outputs to.
    """
    # Setting the timespan to simulate
    timeline = np.arange(2000,tMax,0.1)
    
    # Importing and filtering data
    data = fun.Importer(filename)
    
    # Saving description value
    indicator_text = data["SeriesDescription"][1]
    
    # Filtering data
    obs1 = fun.FilterMaker(data, country1, indicator)
    obs2 = fun.FilterMaker(data, country2, indicator)
    
    # Running a Sanity Check
    print(fun.SanChecker(obs1, obs2))
    
    # Calibrating and simulating a model for both countries
    model1 = fun.calibration(obs1["TimePeriod"].tolist(), obs1["Value"].tolist(), stable=True)
    sim1 = fun.logistic(timeline, model1[0], model1[1], model1[2], model1[3])
    
    model2 = fun.calibration(obs2["TimePeriod"].tolist(), obs2["Value"].tolist(), stable=False)
    sim2 = fun.logistic(timeline, model2[0], model2[1], model2[2], model2[3])
    
    # Storing growth rates and indicator values for chosen year
    growth1 = model1[3]
    growth2 = model2[3]
    
    val2030_1 = sim1[-1]
    val2030_2 = sim2[-1]
    
    # Plotting the simulations versus the observations
    fun.Plotter(obs1, sim1, obs2, sim2, country1, country2, timeline, tMax)
    
    # Adding the simulated values to the DataFrame 
    # This is to allow for evaluation using test statistics.
    obs1["SimVal"] = fun.logistic(obs1["TimePeriod"], model1[0], model1[1], model1[2], model1[3])
    obs2["SimVal"] = fun.logistic(obs2["TimePeriod"], model2[0], model2[1], model2[2], model2[3])
    
    # Evaluating the single-through model.
    full_rsquare1, full_nrmse1, full_bias1 = fun.Evaluator(obs1)
    full_rsquare2, full_nrmse2, full_bias2 = fun.Evaluator(obs1)
    
    # Recreating models using k-fold validation and returning averaged test statistics.
    fold_rsquare1, fold_nrmse1, fold_bias1 = fun.KValidator(obs1, folds)
    fold_rsquare2, fold_nrmse2, fold_bias2 = fun.KValidator(obs2, folds)
    
    # Printing 
    fun.Writer(indicator, indicator_text, country1, country2, growth1, growth2, tMax, val2030_1, val2030_2, \
    full_rsquare1, full_rsquare2, full_nrmse1, full_nrmse2, full_bias1, full_bias2, folds, fold_rsquare1, \
    fold_rsquare2, fold_nrmse1, fold_nrmse2, fold_bias1, fold_bias2, write_file)

    

#%% Running Wrapper

Wrapper("data.csv", "China", "India", "Prevalence of undernourishment (%)", 2030, 5, "output.txt")



# %%

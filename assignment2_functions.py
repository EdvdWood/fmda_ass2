#%% Importing
import pandas as pd
import numpy as np
import assignment2_model as given
import matplotlib.pyplot as plt
import scipy.stats as stats

#%% Function Definitions

# Imports the file and only useful columns
def Importer(filename):
    data = pd.read_csv(filename, skip_blank_lines=True, \
        usecols=["Indicator", "SeriesDescription", "GeoAreaName", "TimePeriod", "Value"])
    return data

# Filtering for the data we need
def FilterMaker(data, country, indicator):
    # Filtering
    data = data[(data["GeoAreaName"]==country) & (data["SeriesDescription"]==indicator)]
    # Making sure we don't get any type errors later
    data = data.astype({"TimePeriod": int, "Value": float})
    data = data.reset_index()
    return data

# Sanity Check that Time Periods for th countries match 
def SanChecker(df1, df2):
    """
        Data Checking Function.
        
        This function makes sure that two pandas DataFrames contain and \
        match on specific columns.
        
        Args:
            df1 (pandas.DataFrame): The first DataFrame to match.
            df2 (pandas.DataFrame): The second DataFrame to match.
            
        Returns:
            output: A string describing the result of the two checks.
    """
    
    if df1['TimePeriod'].equals(df2['TimePeriod']):
        output1 = "Time periods match!"
    else:
        raise Exception("Time period mismatch detected!")
         
    if df1["Value"].isnull().values.any() == False & df2["Value"].isnull().values.any() == False:
        output2 = "No missing values!"
    else: 
        raise Exception("Missing values detected!")
    output = "{0} \n{1}".format(output1, output2)
    return output

# Plotting function
def Plotter(df1, sim1, df2, sim2, country1, country2, timeline, tMax):
    # Setting up Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Plotting Observations
    ax1.scatter(df1["TimePeriod"], df1["Value"], s=10, \
        c='b', marker="s", label=country1)
    ax1.scatter(df2["TimePeriod"], df2["Value"], s=10, \
        c='r', marker="o", label=country2)
    # Plotting Simulations
    ax1.plot(timeline, sim1, c="b", label="{0}, Predicted".format(country1))
    ax1.plot(timeline, sim2, c="r", label="{0}, Predicted".format(country2))
    # Fixing x-axis
    ax1.set_xlim(2000, tMax)
    # Setting labels
    ax1.set(xlabel = "Year", ylabel=df1["SeriesDescription"].iloc[1])
    # Moving Legend out of the way of the data
    plt.legend(loc='upper right')
    # Saving plot
    plt.savefig("EJvdHout_212_China_India.pdf", bbox_inches = "tight")
    plt.close()

# Evaluation Functions    
def NMRSEChecker(sim, obs):
    """
        Normalized Mean Root Squared Error (NRMSE) calculation function.
        
        This function calculates the NMRSE using the statistical \
        function provided in Lecture 7.2. It loops through two arrays \
        of equal length and compares their values.
        
        Args:
            sim (array-like): The simulated array that needs to be checked against observations.
            obs (array-like): The array of observations that is to be checked against.
        
        Returns:
            float:  Normalized Mean Root Squared Error of the simulated array.
    """
    
    midval = []
    for i in range(len(obs)):
        midval.append((sim[i]-obs[i])**2)
    root = np.sqrt(sum(midval)/len(midval))
    normroot = root / np.std(obs)
    return normroot

def PBiasChecker(sim, obs):
    """
        Percentage Bias calculation function.
        
        This function calculates the percentage bias using the statistical \
        function provided in Lecture 7.2. It loops through two arrays \
        of equal length and compares their values using this function.
        
        Args:
            sim (array-like): The simulated array that needs to be checked against observations.
            obs (array-like): The array of observations that is to be checked against.
        
        Returns:
            float:  Percentage Bias of the simulated array.
    """
    
    midval = []
    for i in range(len(obs)):
        midval.append(sim[i]-obs[i])
    pbias = 100 * (sum(midval)/sum(obs))
    return pbias

# Grouping the Evaluation Functions for:

# Single model evaluation
def Evaluator(df):
    """
        Evaluation Function
        
        This function is a simple wrapper for the different evaluation functions available.
        
        Args:
            df (pandas.DataFrame): The DataFrame containing both the simulation and observation data.
            
        Returns:
            Rsquare (float): The R2 value of the simulation compared to the observations.
            NRMSE (float): The NMRSE value of the simulation compared to the observations.
            Bias (float): The Percent Bias of the simulation compared to the observations.
    """
    
    Rsquare = stats.pearsonr(df["SimVal"], df["Value"])[0]**2
    NMRSE = NMRSEChecker(df["SimVal"].tolist(), df["Value"].tolist())
    Bias = PBiasChecker(df["SimVal"].tolist(), df["Value"].tolist())
    return Rsquare, NMRSE, Bias

# K-fold validation
def KValidator(obs, folds):
    """
        K-folds Validation function
        
        This function runs and validates models according to the K-folds method. \
        Every fold model is validated for R2, NRMSE and Percentage Bias. 
        These are then averaged to get the final score for each test statistic.
        
        Args:
            obs (pandas.DataFrame): The DataFrame on which to base \
            the logarithmic model to validate.
            folds (integer): The number of folds to validate with.
        
        Returns:
            Rsquared_ave (float): The averaged R2 value of the different models.
            NRMSE_ave (float): The averaged NMRSE of the different models.
            biases_ave (float): The averaged Percentage Bias of the different models.
    """
    
    # Checking exception: Folds have to contain at least 
    # two values for the Pearson test to function.
    assert(len(obs)>=folds*2)
    # Setting up folds. 
    # Pd.sample(frac=1) is used to shuffle the DataFrame before assigning folds.
    # Counter is used to assign every row a fold. 
    # - Lower folds are larger if no. rows is indivisible by no. folds.
    obs = obs.sample(frac=1)
    counter = 1
    obs["Fold"] = 0
    for row in range(len(obs)):
        # iloc[:,-1] always gets the fold column because it was added last. 
        # This avoids accidentally setting on a copy.
        obs.iloc[row, -1] = counter
        counter += 1
        # Counter is capped at the number of folds.
        if counter > folds: counter = 1
    
    rsquares = []
    nrmses = []
    biases = []
    
    # Run Validator
    for fold in range(1,folds+1):
        train = obs[obs["Fold"]!=fold].copy()
        test = obs[obs["Fold"]==fold].copy()
        model = calibration(np.array(train["TimePeriod"]), np.array(train["Value"]))
        train_out = logistic(test["TimePeriod"].tolist(), model[0], model[1], model[2], model[3])
        rsquares.append(stats.pearsonr(train_out, test["Value"])[0]**2)
        nrmses.append(NMRSEChecker(train_out, test["Value"].tolist()))
        biases.append(PBiasChecker(train_out, test["Value"].tolist()))
        
        rsquare_ave = np.mean(rsquares)
        nmrse_ave = np.mean(nrmses)
        bias_ave = np.mean(biases)
    return rsquare_ave, nmrse_ave, bias_ave

# Generating and writing output
def Writer(indicator, indicator_text, country1, country2, growth1, growth2, tMax, val2030_1, val2030_2, \
    full_rsquare1, full_rsquare2, full_nrmse1, full_nrmse2, full_bias1, full_bias2, folds, fold_rsquare1, \
    fold_rsquare2, fold_nrmse1, fold_nrmse2, fold_bias1, fold_bias2, write_file):
    
    # Initialize output holder
    outputs = []
    
    # Set up the output
    outputs.append("The Selected SDG Indicator is Indicator {0}: {1}\n\n".format(indicator, indicator_text))
    outputs.append("Country\t\t\t\t{0}\t{1}\n".format(country1, country2))
    outputs.append("HDI:\t\t\t\t0.758\t0.647\n") #This would have to be in a datafile to make the code more flexible.
    outputs.append("Growth Value:\t\t\t{0:.2f}\t{1:.2f}\n".format(growth1, growth2))
    outputs.append("Indicator in {0}:\t\t{1:.3f}\t{2:.3f}\n".format(tMax, val2030_1, val2030_2))
    outputs.append("Model Evaluation R2:\t\t{0:.3f}\t{1:.3f}\n".format(full_rsquare1, full_rsquare2))
    outputs.append("Model Evaluation NMRSE:\t\t{0:.3f}\t{1:.3f}\n".format(full_nrmse1, full_nrmse2))
    outputs.append("Model Evaluation PBIAS (%):\t{0:.3f}\t{1:.3f}\n".format(full_bias1, full_bias2))
    outputs.append("{0}-fold Model Validation R2:\t{1:.3f}\t{2:.3f}\n".format(folds, fold_rsquare1, fold_rsquare2))
    outputs.append("{0}-fold Model Validation NRMSE:\t{1:.3f}\t{2:.3f}\n".format(folds, fold_nrmse1, fold_nrmse2))
    outputs.append("{0}-fold Model Validation PBIAS (%):{1:.3f}\t{2:.3f}\n".format(folds, fold_bias1, fold_bias2))
    
    # Create File:
    with open(write_file, 'w+') as newfile:
    # Loop through and print lines
        for line in outputs:
            newfile.write(line)

            
#%% Given Functions

# logistic model
def logistic(x, start, K, x_peak, r):
    """
    Logistic model
    
    This function runs a logistic model.
    
    Args:
        x (array_like): The control variable as a sequence of numeric values \
        in a list or a numpy array.
        start (float): The initial value of the return variable.
        K (float): The carrying capacity.
        x_peak (float): The x-value with the steepest growth.
        r (float): The growth rate.
        
    Returns:
        array_like: A numpy array or a single floating-point number with \
        the return variable.
    """
    
    if isinstance(x, list):
        x = np.array(x)
    return start + K / (1 + np.exp(r * (x_peak-x)))

def calibration(x, y, stable = False):
    """
    Calibration
    
    This function calibrates a logistic model.
    The logistic model can have a positive or negative growth.
    
    Args:
        x (array_like): The explanatory variable as a sequence of numeric values \
        in a list or a numpy array.
        y (array_like): The response variable as a sequence of numeric values \
        in a list or a numpy array.
        stable (bool, optional): Indication if the logistic growth already \
        slowed down and stabilized or not (default: False).
        
    Returns:
        tuple: A tuple including four values: 1) the initial value (start), \
    2) the carrying capacity (K), 3) the x-value with the steepest growth \
    (x_peak), and 4) the growth rate (r).
    """
    
    slope = [None] * (len(x) - 1)
    for i in range(len(slope)):
        slope[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        slope[i] = abs(slope[i])
    x_peak = x[slope.index(max(slope))] + 0.5
    
    if stable:
        start = min(y)
        K = max(y) - start
    else:
        if y[0] < y[-1]: # positive growth
            start = min(y)
            K = 2 * (sum([y[slope.index(max(slope))], \
                            y[slope.index(max(slope))+1]])/2 - start)
        else: # negative growth
            K = 2 * (max(y) - sum([y[slope.index(max(slope))], \
                            y[slope.index(max(slope))+1]])/2)
            start = max(y) - K
    
    performance = [None] * 1000
    if y[0] < y[-1]: # positive growth
        r = np.arange(0.01, 10.01, 0.01)
    else: # negative growth
        r = np.arange(-0.01, -10.01, -0.01)
    for i in range(len(r)):
        model = logistic(x, start, K, x_peak, r[i])
        performance[i] = NMRSEChecker(model, y)
    r = r[performance.index(min(performance))]
    return start, K, x_peak, r





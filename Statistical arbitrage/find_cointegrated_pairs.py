from cointegration_analysis import estimate_long_run_short_run_relationships, engle_granger_two_step_cointegration_test
import pandas as pd
import numpy as np

# Find cointegrated pairs from historical data to be used for developing a statistical arbitrage strategy

data = pd.read_csv('cointegration/data.csv') # data file can be found in the master branch
data.drop("Unnamed: 0", axis=1 , inplace = True)
cointegrated_pairs = [] 

for s1 in data.columns: # Find cointegrated pairs based on data
    for s2 in data.columns:
        if s1 != s2: # Note that the pairs (s1,s2) and (s2,s1) do not necessarily yield the same results so we consider them seperately
            _ , p_val = engle_granger_two_step_cointegration_test(np.log(data[s1]), np.log(data[s2]))
                    
            if p_val < 0.01:
                cointegrated_pairs.append((s1, s2))
        
    
cs = {}
gammas = {}
alphas = {}
stds = {}
    
# Calculate required parameters for these cointegrated pairs

for pair in cointegrated_pairs:
    
    y = np.log(data[pair[0]])
    x = np.log(data[pair[1]])
    c, gamma, alpha , z = estimate_long_run_short_run_relationships(y, x)
    gammas[pair] = gamma
    alphas[pair] = alpha
    stds[pair] = np.std(z)
    cs[pair] = c
    
    

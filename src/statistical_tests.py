import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def run_spatiotemporal_diagnostic(df):
    """
    Performs a Chi-Square Test to prove that crime is not random. 
    Returns the p-value and Cramer's V effect size.
    """
    # 1. Create a Contingency Table (Area vs Time Bucket)
    # This checks if certain areas have 'signature' crime times
    contingency_table = pd.crosstab(df['AREA NAME'], df['time_bucket'])
    
    # 2. Run the Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # 3. Calculate Cramer's V (Effect Size)
    # Values > 0.1 indicate a strong, predictable pattern
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    return {
        "chi2": chi2,
        "p_value": p,
        "cramers_v": cramers_v,
        "is_significant": p < 0.05
    }
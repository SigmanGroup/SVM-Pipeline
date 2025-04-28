import numpy as np

def assess_collinearity(dataframe, threshold):
    corr_matrix = np.abs(dataframe.corr()) 
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  
    collinear_features = [column for column in upper_triangle.columns if any(
        upper_triangle[column] > threshold)]  
    return collinear_features
# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

data = pd.read_csv('ICR - dataset/train.csv')
data.shape

# # # # # # # # # # # # # # # #
# # # # # Preprocessing # # # #
# # # # # # # # # # # # # # # #

## stage 1 ##
# filling up missing data
print(data.columns[data.isnull().sum() > 0])

# to minimize the effect of the filling data,
# we've deceided to fill in all empty cells by the mean of the column
columns_has_empty = ['BQ', 'CB', 'CC', 'DU', 'EL', 'FC', 'FL', 'FS', 'GL']
data[columns_has_empty] = data[columns_has_empty].fillna(data[columns_has_empty].mean())

# validate the completeness of filling
print(data.columns[data.isnull().sum() > 0])

## stage 2 ##
# hot encoding the binary column 'EJ' by (A = 0, B = 1)
data.loc[data['EJ'] == 'A', 'EJ'] = 1
data.loc[data['EJ'] == 'B', 'EJ'] = 0

features_names = ['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN',
       'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS',
       'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY',
       'EB', 'EE', 'EG', 'EH', 'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI',
       'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL']
target_name = 'Class'

# %%
## stage 3 ##
# dealing with outliers
# build a zero matrix used to record outliers
matrix = np.zeros((data.shape[0], data.shape[1]), dtype = object)
matrix[:, 0] = data['Id'].tolist()

def find_outliers(data):
    # Step 1: Calculate Q1, Q3, and IQR
    # Sort the data based on the second column (index 1)
    sorted_data = data[data[:, 1].argsort()]

    q1 = np.percentile(sorted_data[:, 1], 25)
    q3 = np.percentile(sorted_data[:, 1], 75)
    iqr = q3 - q1

    # Step 2: Find lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Step 3: Identify outliers in the dataset
    outliers = sorted_data[(sorted_data[:, 1] < lower_bound) | (sorted_data[:, 1] > upper_bound)]
    
    return outliers

# record all outliers as 1 in zero matrix
i = 0
for index, features in enumerate(data.columns.tolist()):
    if features  != 'Id' and features  != 'EJ' and features  != 'Class':
        outliers = find_outliers(data.iloc[:, [0, i]].values)
        
        j = 0
        for id in matrix[:, 0]:
            if id in outliers[:, 0]:
                matrix[j, i] = 1
            j += 1

    i += 1

# compute the number of features are outlier for each observation
num_outliers = []
for entry in matrix:
    entry_sum = np.sum(entry[1:])
    num_outliers.append(entry_sum)
    
matrix = np.column_stack((matrix, num_outliers))

# find these observations have too many outlier features and remove them
# Train = pd.concat([data, pd.DataFrame(data, columns = ['Class'])], axis = 1)

outliers = find_outliers(matrix[:, [0, -1]])
outliers = pd.DataFrame(outliers[:, 0], columns=['Id'])
data = data[~data['Id'].isin(outliers.iloc[:, 0])]

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# %%
## stage 4 ##
# standardization train data
numeric_cols = [features for index, features in enumerate(X.columns.tolist()) if(features != 'Class' and features != 'EJ')]
sc = StandardScaler()
X[numeric_cols] = sc.fit_transform(X[numeric_cols])



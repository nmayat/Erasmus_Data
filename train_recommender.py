# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 23:20:24 2023

@author: nils
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def process_test_data_in_batches(X_test, y_test, model, batch_size=10):
    reciprocal_ranks = []
    num_batches = int(np.ceil(X_test.shape[0] / batch_size))

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, X_test.shape[0])

        X_test_batch = X_test[start_idx:end_idx]
        y_test_batch = y_test.iloc[start_idx:end_idx]

        distances, indices = model.kneighbors(X_test_batch)
        
    
        y_pred_batch = [y_train.iloc[indices_row].values for indices_row in indices]
        y_true_batch = y_test_batch.values

        # Calculate reciprocal rank for each instance in the batch
        batch_reciprocal_ranks = [
            1 / (np.where(y_pred == y_true)[0][0] + 1)
            if y_true in y_pred else 0
            for y_true, y_pred in zip(y_true_batch, y_pred_batch)
        ]

        reciprocal_ranks.extend(batch_reciprocal_ranks)
    return np.mean(reciprocal_ranks)

data = pd.read_pickle('Erasmus_Data/model_data.pkl')



# Label encode the 'features' and 'Receiving Country' columns
encoder = LabelEncoder()
categorical_columns = ['Activity', 'Field of Education', 'Education Level', 'Sending Country', 'Receiving Country']
encoded_df = data.copy()
for column in categorical_columns:
    encoded_df[column] = encoder.fit_transform(data[column])

cat_transformer = OneHotEncoder(handle_unknown='ignore')
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ['Mobility Duration', 'Participant Age']),
        ('cat', cat_transformer, ['Activity', 'Field of Education', 'Education Level', 'Sending Country'])
    ],sparse_threshold=0)

# Fit and transform the data
X = encoded_df.drop('Receiving Country', axis=1)
y = encoded_df['Receiving Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
# Define the model
model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric='euclidean')
# Train the model on the training set
model.fit(X_train)

# Get the column names after preprocessing
num_columns = ['Mobility Duration', 'Participant Age']
cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns[:-1])

# Combine the column names
transformed_columns = np.concatenate([num_columns, cat_columns])
X_test_df = pd.DataFrame(X_test, columns=transformed_columns)

#average_mrr = process_test_data_in_batches(X_test, y_test, model, batch_size=2000)

#print("Average MRR: {:.4f}".format(average_mrr))


def recommend_receiving_country(student_info):
    # Encode the student_info
    encoded_student_info = student_info.copy()
    for i, column in enumerate(categorical_columns[:-1]):
        encoded_student_info[column] = encoder.fit(data[column]).transform([student_info[column]])[0]

    # Transform the student_info
    student_X = preprocessor.transform(encoded_student_info)

    # Find the k-nearest neighbors
    distances, indices = model.kneighbors(student_X)

    # Get the corresponding receiving countries
    recommendations = data.iloc[indices[0]]['Receiving Country'].values

    return np.unique(recommendations)

# Example usage
student_info = pd.DataFrame([{'Mobility Duration': 90, 'Activity': 'Individual Volunteering Activities', 'Field of Education': 'Fine arts',
                              'Education Level': 'Third cycle / Doctoral or equivalent level', 'Participant Age': 35,
                              'Sending Country': 'Lesotho'}])

recommendations = recommend_receiving_country(student_info)
print(recommendations)

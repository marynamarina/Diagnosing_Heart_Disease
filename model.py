# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

DF_URL = 'https://raw.githubusercontent.com/devrepublik/data-science-course/master/data/classification/HeartDisease_Clean.csv'
stroke_df = pd.read_csv(DF_URL)

X_train, X_test, y_train, y_test = train_test_split(
    stroke_df.drop('target', axis=1),
    stroke_df['target'],
    test_size=0.1,
    random_state=42)


regressor = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    max_samples=0.5,
    oob_score=True,
    random_state=42)

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
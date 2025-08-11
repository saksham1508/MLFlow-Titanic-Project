# Utility functions for Titanic project

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_missing_values(df):
	"""
	Fill missing values in the Titanic dataset.
	Age: fill with median
	Embarked: fill with mode
	Fare: fill with median
	Cabin: fill with 'Unknown'
	"""
	df['Age'] = df['Age'].fillna(df['Age'].median())
	df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
	df['Fare'] = df['Fare'].fillna(df['Fare'].median())
	df['Cabin'] = df['Cabin'].fillna('Unknown')
	return df

def encode_features(df):
	"""
	Encode categorical features: Sex, Embarked, Cabin (first letter only)
	"""
	df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
	df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
	# Optionally, use only the first letter of Cabin
	df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
	df['Cabin'] = LabelEncoder().fit_transform(df['Cabin'])
	return df

def add_family_size(df):
	"""
	Add a FamilySize feature (SibSp + Parch + 1)
	"""
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	return df

def preprocess_titanic(df):
	"""
	Complete preprocessing pipeline for Titanic dataset.
	"""
	df = fill_missing_values(df)
	df = add_family_size(df)
	df = encode_features(df)
	return df


"""
personality_classifier.py
This script trains a model to idenditify an 'Introvert' or an 'Extrovert' using a Random Forest Classifier and pipeline based 
preprocessing. Cross-validation, grid search and evaluation metrics are also included.  """

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
from joblib import dump
from pathlib import Path



 
##  Detects root level of code to ensure compatibility  
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUALS_DIR = PROJECT_ROOT / "visuals"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"


def main():

  raw_df = pd.read_csv(DATA_DIR/ r'C:\Users\jolie\OneDrive\Desktop\Python Code\GitHub Projects\personality-classifier\data\personality_dataset.csv')

## Creating duplicate of datafram (df) to avoid changing the original data
  data = raw_df.copy()


## Choosing target column and feature columns
  y = data.Personality
  X = data.drop('Personality', axis = 1)


## Checks for missing values in all columns
  missing = data.isnull().sum()
  missing_features = missing[missing > 0]
  missing_percent = (missing_features.values)/ len(data) * 100


## Bar plot of percentage of missing values per column
  sns.barplot(x= missing_features.index, y = missing_percent, palette="viridis" )
  plt.title('Percentage of Missing Values Per Column')
  plt.xlabel("Columns")
  plt.ylabel("Percentage of Missing Values")
  plt.xticks(rotation= 80)
  plt.tight_layout()
  plt.savefig(VISUALS_DIR/ "bar_plot_missing_values.png")
  plt.close()


## Label encoding target (y) values to 
  lab_en = LabelEncoder()
  y_copy = y.copy()   # makes copy of original y data so it is not overwritten
  y_encoded = lab_en.fit_transform(y_copy)


## Preprocessing for X data (features)
# First find object columns so that they can be preprocessed separately.
  object_cols = [col for col in X if X[col].dtypes == 'object']


# Find numerical X data columns so that they can be preprocessed separately.
  num_cols = [col for col in X if X[col].dtypes in ['float64', 'int64']]


# # Create pipeline for encoding and imputing preprocesses for X data 
  categorical_X_processes = Pipeline(steps =[('ord_enc', OrdinalEncoder()), ('impute', SimpleImputer(strategy= 'most_frequent'))])
  numerical_X_processes = Pipeline(steps =[('impute', SimpleImputer(strategy = 'mean'))])


## Applying Pipelines to specific columns 
# Column Transformer                   
  preprocess = ColumnTransformer(transformers = [('X_cat_data', categorical_X_processes, object_cols), ('X_num_data', numerical_X_processes, num_cols)])


# Make model
  personality_model = RandomForestClassifier(n_estimators= 100, random_state= 42) 

# Create main pipeline 
  my_pipeline = Pipeline(steps=[('preprocess', preprocess), ('model', personality_model)])


## Cross validate
# KFold setup
  kFold = KFold(n_splits= 5, shuffle= True, random_state= 42)

## Baseline score validation for initial check (optional)                                                       
# results = cross_val_score(my_pipeline, X, y_encoded, cv= kFold, scoring= 'accuracy')
# print('Cross validation scores:', results)
# print('Average score:', results.mean())


### Fine Tuning
  param_grid= {'model__n_estimators' : [ 100, 200, 300 ,400], 'model__max_depth': [None, 5, 10, 15 ,20] }
  scoring_metrics = {'accuracy': 'accuracy', 'f1_macro':'f1_macro'}
  grid_search = GridSearchCV(my_pipeline, param_grid, cv=kFold, refit= 'f1_macro', scoring= scoring_metrics, return_train_score=True )
  grid_search.fit(X, y_encoded)
  print('Best F1 Macro Score:', grid_search.best_score_)
  print('Best Parameters:', grid_search.best_params_)


### Plot scores  
  results_df = pd.DataFrame(grid_search.cv_results_)
### Saves results 
  results_df.to_csv(MODEL_DIR/ "grid_search_results.csv", index=False)
#   print(results_df[['params', 'mean_test_accuracy', 'mean_test_f1_macro']])


  sns.lineplot(x=results_df['param_model__n_estimators'], y=results_df['mean_test_f1_macro'], label='F1_macro Score')
  sns.lineplot(x=results_df['param_model__n_estimators'], y=results_df['mean_test_accuracy'], label='Accuracy')
  plt.xlabel('n_estimators')
  plt.ylabel('Score')
  plt.title('Model Performance vs. n_estimators')
  plt.legend()
  plt.tight_layout()
  plt.savefig(VISUALS_DIR/ "grid_search_scores.png")
  plt.close()


## Classification report
  y_pred = cross_val_predict(my_pipeline, X, y_encoded, cv=kFold)

## Generate classification report
  print("Classification report:" ,classification_report(y_encoded, y_pred, target_names=lab_en.classes_))

## Plot confusion matrix
  ConfusionMatrixDisplay.from_predictions(y_encoded, y_pred, display_labels=lab_en.classes_)
  plt.title("KFold Cross-Validated Confusion Matrix")
  plt.tight_layout()
  plt.savefig(VISUALS_DIR/ "confusion_matrix.png")
  plt.close()


## Final model with optimized parameters fit to full data set
  best_model = grid_search.best_estimator_
  best_model.fit(X, y_encoded)

  dump(best_model, MODEL_DIR/ "personality_classifier.pkl")



if __name__ == "__main__":
    main()
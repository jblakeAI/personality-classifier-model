<<<<<<< HEAD

## Personality Classifier

This mini project uses a machine learning model to classify individuals as *introverts* or *extroverts* based on personality traits and the duration of social activities they engage in. 
As a first-time machine learning model builder, this felt like a good place to start tinkering. 


# Project Overview
- **Dataset**: Personality data (CSV format)
    link : https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data
- **Goal**: Predict binary personality type (`Introvert` or `Extrovert`)
- **Model**: Random Forest Classifier
- **Tools**: Python, scikit-learn, pandas, seaborn, matplotlib


# Project Strucutre
personality-classifier/

├── data/ # Raw dataset (.csv)

├── model/ # Saved trained model (.pkl)

├── visuals/ # Generated plots and figures

├── src/ # Main ML pipeline script

│ └── model_pipeline.py

├── README.md

├── .gitignore

├── requirements (.txt)


# Requirements
Key libraries:

pandas

scikit-learn

seaborn

matplotlib

joblib

**Full list in requirements.txt.**


# How to run
Follow the steps below to setup and run the personality classifier:
1. Clone the repository
   
   git clone https://github.com/jblakeAI/personality-classifier-model.git

   cd personality-classifier-model

3. Set up virtual environment

   python -m venv venv

4. Activate it
 
    **On Windows**: venv\Scripts\activate 

   **On Mac/Linux**: source venv/bin/activate  

5. Install dependencies: 
   pip install -r requirements.txt 

6. Run the machine learning pipeline from the root not the src/
   python src/model_pipeline.py
   

**This script will then peform the following steps:**

- Preprocess the dataset

- Train and evaluate a Random Forest classifier

- Generate plots in the visuals/ directory

- Save the trained model to the model/ directory

- Display a classification report and confusion matrix



# Output structure
- Bar chart of percentage of missing values per column
- Grid search results
- Line plot of F1 macro and accuracy score
- Classification report and confusion matrix
- Saved trained model: model/personality_classifier.pkl



# Model Performance
Best F1 Macro Score: 0.9341972054024634

Cross-validated using 5-fold KFold with grid search.

# License
This project is licensed under the MIT License.







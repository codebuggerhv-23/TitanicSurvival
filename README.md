🚢 Titanic Survival Prediction — Machine Learning Project
A data science project predicting passenger survival on the RMS Titanic using supervised machine learning.
This project demonstrates an end-to-end ML workflow, including preprocessing, analysis, feature engineering, scaling, regularization, and evaluation — all implemented in Python using Pandas, NumPy, Seaborn, and Scikit-learn.

📘 1. Project Overview
The goal of this project is to analyze Titanic passenger data, extract meaningful features, and build a predictive model that estimates the likelihood of survival based on demographic and travel-related attributes.

This serves as a classic example of a binary classification problem in machine learning.

🎯 2. Objectives
Perform complete data cleaning and exploratory analysis on the Titanic dataset.
Apply feature engineering and scaling for better model performance.
Train and optimize a classification model using regularization techniques.
Interpret the model and identify influential features affecting survival.
🧩 3. Dataset Description
Source: Kaggle — Titanic: Machine Learning from Disaster

Target Variable:
Survived → 1 = Survived, 0 = Did not survive

Key Features:

Column	Description
PassengerId	Unique identifier for each passenger
Pclass	Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
Name	Passenger name
Sex	Gender
Age	Age in years
SibSp	Number of siblings/spouses aboard
Parch	Number of parents/children aboard
Ticket	Ticket number
Fare	Passenger fare
Cabin	Cabin number
Embarked	Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
⚙️ 4. Workflow Summary
🧹 Step 1: Data Preprocessing
Inspected and handled missing values.
Filled missing Age with mean, Embarked with mode.
Dropped unnecessary columns (PassengerId, Name, Ticket, Cabin).
Encoded categorical features into numeric form.
📊 Step 2: Data Analysis
Explored survival rates by gender, class, embarkation port, and age distribution.
Visualized survival trends using Seaborn countplots and barplots.
Key Findings:

Females had higher survival probability.
1st-class passengers had better survival rates.
Traveling alone decreased survival likelihood.
🧱 Step 3: Data Preparation
Split data into train (80%) and test (20%) using train_test_split().
Ensured inclusion of only relevant and encoded features.
🤖 Step 4: Baseline Model
Built initial Logistic Regression model.
Evaluated accuracy, confusion matrix, and classification report.
Detected minor overfitting → applied regularization.
🧬 Step 5: Feature Engineering
Created new derived features:

FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1, else 0
Title = extracted from the “Name” column (Mr, Mrs, Miss, etc.)
These features enhanced interpretability and model performance.

📏 Step 6: Feature Scaling
Used StandardScaler to normalize:

Age, Fare, FamilySize
Ensured training mean ≈ 0 and std ≈ 1.

🧮 Step 7: Model Training with Regularization
Implemented Logistic Regression with:

Ridge (L2) and Lasso (L1) regularization
Tuned C values → [0.01, 0.1, 1, 10, 100]
Best performance:
C = 0.1 with Ridge (L2) — minimized overfitting, maximized accuracy.

Regularization	C	Train Accuracy	Test Accuracy
Ridge (L2)	0.1	~0.82	~0.77
Lasso (L1)	0.1	~0.81	~0.76
🧾 Step 8: Final Result & Interpretation
Final Model: Logistic Regression (L2 Regularization, C=0.1)

Performance:

Accuracy ≈ 77%
Precision & Recall balanced
Confusion Matrix showed majority correctly classified
Most Influential Features:

🔼 Positive: Sex (female), Title_Mrs, Pclass (1st), Fare (high)
🔽 Negative: IsAlone, Age, FamilySize (large)
💡 5. Key Takeaways
Preprocessing, feature engineering, and regularization significantly improved model performance.
Ridge regularization balanced bias and variance well.
Logistic Regression coefficients provided interpretability for survival factors.
🧰 6. Tools and Technologies
Language: Python 3.10+

Libraries Used:

pandas, numpy — data handling
matplotlib, seaborn — visualization
scikit-learn — model building, scaling, regularization, evaluation
🚀 7. Future Enhancements
Use ensemble models (RandomForest, XGBoost, GradientBoosting)
Apply cross-validation for stronger generalization
Add ROC-AUC and precision-recall visualization
Deploy using Flask or Streamlit for interactive prediction
👨‍💻 8. Author
Name: Dashmeet Singh Malhotra
Role: Machine Learning Enthusiast / ML Intern Applicant

Objective: To explore real-world ML workflows, build predictive systems, and develop interpretable AI models.

📬 Contact:

Email: [dashcodeworks@gmail.com]
LinkedIn: [https://www.linkedin.com/in/dashmeet-singh-malhotra-6a90a3325/]
GitHub: [https://github.com/Dashmeet-S-Malhotra]

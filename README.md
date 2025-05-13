💳 Credit Card Fraud Detection using Neural Networks
This project focuses on detecting fraudulent credit card transactions using machine learning techniques, specifically Artificial Neural Networks (ANNs) implemented in Python. The dataset used is sourced from Kaggle, and it contains anonymized credit card transactions made by European cardholders in September 2013.


📊 Dataset Description
Source: Kaggle - Credit Card Fraud Detection

Samples: 284,807 transactions

Fraudulent Transactions: 492

Features: 30 anonymized features (V1-V28), Time, Amount, and Class (0 = legit, 1 = fraud)

🧠 Model Architecture
Input Layer: 30 features

Hidden Layers: 2–3 Dense layers with ReLU activation

Output Layer: 1 neuron with Sigmoid activation (binary classification)

Loss Function: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall, F1 Score, AUC

🚀 How to Run
1. Clone this repository
bash
Copy
Edit
git clone https://github.com/7vik-dev/Creditcard-Fraud-detection-.git
cd credit-card-fraud-detection
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If no requirements.txt is available, manually install:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
3. Run the notebook
Launch Jupyter Notebook or use:

bash
Copy
Edit
jupyter notebook Credit_Card_Fraud_Detection.ipynb
📈 Evaluation Metrics
Due to data imbalance, accuracy is not sufficient. This project emphasizes:

Confusion Matrix

Precision-Recall Curve

ROC-AUC Score

F1 Score

⚖️ Handling Imbalanced Data
This project uses:

Undersampling / Oversampling (if needed)

Class Weights

SMOTE (Synthetic Minority Oversampling Technique) (optional, to be explored)

🛠 Future Improvements
Experiment with other ML models (Random Forest, XGBoost)

Use advanced sampling techniques (SMOTE, ADASYN)

Hyperparameter tuning with Keras Tuner or GridSearchCV

Deploy the model via Flask API

📜 License
This project is open-source and available under the MIT License.

🙌 Acknowledgements
Dataset by ULB Machine Learning Group via Kaggle

TensorFlow and Scikit-learn documentation

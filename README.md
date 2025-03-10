RetenX - Employee Attrition Prediction System

📌 Project Overview
RetenX is a Machine Learning-based Employee Attrition Prediction System designed to help organizations analyze employee retention trends. It uses multiple ML algorithms to predict which employees are at risk of leaving, enabling businesses to take proactive measures.


Features
✅ Predict employee attrition using 5 ML models (Random Forest, Logistic Regression, SVM, KNN, XGBoost).

✅ Handle datasets with null values for better accuracy.

✅ User-friendly Flask web interface for predictions and analysis.

✅ Clean and modern UI inspired by x.ai.

✅ Comparison of different ML models' performance.

✅ Dataset processing for bulk employee attrition prediction.

✅ Historical trends & retention strategy suggestions based on insights.


Project Structure

RetenX/
│── datasets/
│── models/                 
│── templates/              
│── static/                 
│── training/               
│── app.py                  
│── requirements.txt        
│── README.md               
│── .gitignore             

Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/Rubel286/RetenX.git

cd RetenX

2️⃣ Create & Activate Virtual Environment

On Windows:

python -m venv venv

venv\Scripts\activate

On macOS/Linux

python3 -m venv venv

source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Flask App

python app.py

Then, open http://127.0.0.1:5000/ in your browser.

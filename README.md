🧠 Thyroid Cancer Recurrence Prediction

This project is a web-based application built with Django and Machine Learning for predicting the recurrence of thyroid cancer based on clinical and historical patient data. It uses trained models and encoded features to make predictions from user input.

---

## 📌 Features

- 🎯 Predict thyroid cancer recurrence
- 🔐 Encoded clinical data (.pkl files) for efficient prediction
- 📊 Clean, interactive UI built with Django and HTML
- 📁 Modular project structure following Django best practices
- 💾 Trained Machine Learning model for inference

---

## 🛠️ Tech Stack

- **Backend**: Django, Python
- **Frontend**: HTML, Bootstrap (optional)
- **ML**: Scikit-learn, Pandas
- **Model Files**: `.pkl` encoders for categorical features

---

## 📂 Project Structure

THYROID_CANCER/
│
├── predictor/
│ ├── templates/predictor/index.html
│ ├── models.py
│ ├── views.py
│ ├── forms.py
│ └── ... (other standard Django files)
│
├── *.pkl (encoder files for categorical feature transformation)
├── requirements.txt
├── .gitignore
└── manage.py

yaml
Copy
Edit

---

## 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/Nitu143-nks/ThyroidCancerPredictor.git
cd ThyroidCancerPredictor
Create and activate a virtual environment

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Django server

bash
Copy
Edit
python manage.py runserver
Open in Browser

Visit: http://127.0.0.1:8000/
and input the values to see predictions.

🧠 Thyroid ML Model
The model uses multiple .pkl encoders to preprocess inputs and generate recurrence predictions based on patient symptoms and medical history.

📜 License
This project is for academic and demonstration purposes only.

🤝 Author
Nitesh Kumar Sahoo

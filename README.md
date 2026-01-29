🏥 Medical Appointment No-Shows – Streamlit Project

📌 What is this project?
This is a Streamlit application that analyzes the **Medical Appointment No-Shows** dataset to understand why patients miss their scheduled medical appointments.

The project focuses on:

* Exploring the dataset structure and quality
* Handling data quality issues and logical inconsistencies
* Cleaning and preparing the data
* Feature engineering from date and categorical variables
* Scaling numerical features
* Visualizing patterns related to appointment attendance
* Applying PCA for dimensionality reduction and final visualization

📊 Dataset

* Source: Kaggle – Medical Appointment No Shows
* Rows: 110,527
* Target variable: NoShow (Whether the patient missed the appointment)

🛠 Tools Used

* Python
* Pandas & NumPy
* Streamlit
* Plotly & Matplotlib
* Scikit-learn
* OpenCV (basic image preprocessing)

▶️ How to Run the App

```
pip install -r requirements.txt
streamlit run app.py
```
🌐 Live App 👉 (https://medical-appointment-hepyscvw73ewhupk63c5gw.streamlit.app/)

📁 Project Files

```
app.py                 # Streamlit application code
Medical Appointment.csv # Dataset
No Shows.png           # App image
requirements.txt       # Required libraries
README.md              # Project description
```

✅ Summary
This project demonstrates a complete and structured data analysis pipeline using an interactive Streamlit application, focusing on data quality, exploratory analysis, feature engineering, and visualization without applying predictive modeling.

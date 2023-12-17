from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_url_path='/static', static_folder='static')
# Load your machine learning model
pca=joblib.load("C:/Users/adars/Downloads/pca.joblib")
model = joblib.load("C:/Users/adars/Downloads/your_model (1).joblib")

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/')
def home():
    carousel_images = [
        {"src": "/static/img3.png", "alt": "Image 1"},
        {"src": "/static/img4.png", "alt": "Image 2"},
        {"src": "/static/img2.png", "alt": "Image 2"},
        # Add more images as needed
    ]
    return render_template('home.html', carousel_images=carousel_images)

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/about')
def about():
    return render_template('about.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get user input from the form
#     new_instance_input = request.form['new_instance']

#     # Split the input string into a list of values
#     new_instance_values = new_instance_input.split(',')

#     # Convert each value to float, replacing '?' with np.nan
#     new_instance = [float(val) if val != '?' else np.nan for val in new_instance_values]

#     # Impute missing values (replace np.nan with mean)
#     mean_imputer = SimpleImputer(strategy='mean')
#     new_instance_imputed = mean_imputer.fit_transform([new_instance])

#     # Make predictions
#     new_instance_transformed = pca.transform([new_instance])  # Reshape to 2D array
#     prediction = model.predict(new_instance_transformed)

#     # Convert the prediction to a human-readable class name
#     # Add your class mapping here
#     class_mapping = {
#         1: "Normal",
#         2: "Ischemic changes (CAD)",
#         3: "Old Anterior Myocardial Infarction",
#         4: "Old Inferior Myocardial Infarction",
#         5: "Sinus tachycardia",
#         6: "Sinus bradycardia",
#         7: "Ventricular Premature Contraction (PVC)",
#         8: "Supraventricular Premature Contraction",
#         9: "Left Bundle Branch Block",
#         10: "Right Bundle Branch Block",
#         11: "1st Degree Atrioventricular Block",
#         12: "2nd Degree AV Block",
#         13: "3rd Degree AV Block",
#         14: "Left Ventricular Hypertrophy",
#         15: "Atrial Fibrillation or Flutter",
#         16: "Others"
#     }
#     predicted_class_name = class_mapping.get(prediction[0], 'Unknown')

#     return render_template('result.html', prediction=predicted_class_name)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', prediction='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', prediction='No file selected')

    # Load CSV file into a DataFrame
    df = pd.read_csv(file)

    # Impute missing values (replace np.nan with mean)
    mean_imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(mean_imputer.fit_transform(df), columns=df.columns)

    # Make predictions
    df_transformed = pca.transform(df_imputed)
    predictions = model.predict(df_transformed)
    class_mapping = {
        1: "Normal",
        2: "Ischemic changes (CAD)",
        3: "Old Anterior Myocardial Infarction",
        4: "Old Inferior Myocardial Infarction",
        5: "Sinus tachycardia",
        6: "Sinus bradycardia",
        7: "Ventricular Premature Contraction (PVC)",
        8: "Supraventricular Premature Contraction",
        9: "Left Bundle Branch Block",
        10: "Right Bundle Branch Block",
        11: "1st Degree Atrioventricular Block",
        12: "2nd Degree AV Block",
        13: "3rd Degree AV Block",
        14: "Left Ventricular Hypertrophy",
        15: "Atrial Fibrillation or Flutter",
        16: "Others"
    }
    # Convert predictions to human-readable class names
    predicted_class_names = [class_mapping.get(pred, 'Unknown') for pred in predictions]

    return render_template('result.html', predictions=predicted_class_names)


if __name__ == '__main__':
    app.run(debug=True)

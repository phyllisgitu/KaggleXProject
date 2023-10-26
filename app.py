import pickle
import os
import pandas as pd
from flask import Flask, render_template, request
from waitress import serve
current_directory = os.getcwd()
print(f'Current Working Directory: {current_directory}')
app = Flask(__name__, static_url_path='/static', static_folder='static')
model = pickle.load(open('life_insurance_classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    '''Render the template'''
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    '''Upload csv file then Predict using our loaded model'''
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != '':
        # Save the uploaded file to a temporary location
        file_path = f"temp/{uploaded_file.filename}"
        uploaded_file.save(file_path)

        # Read the uploaded CSV file
        sample = pd.read_csv(file_path)
        sample.columns = ['f', 'v']
        selected_f = ['Product_Info_1', 'Product_Info_2', 'Product_Info_4', 'Ins_Age', 'Wt',
                       'BMI', 'Employment_Info_2', 'Employment_Info_4', 'InsuredInfo_1',
                       'InsuredInfo_2', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
                       'Insurance_History_2', 'Insurance_History_9', 'Family_Hist_1',
                       'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_4',
                       'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
                       'Medical_History_11', 'Medical_History_12', 'Medical_History_13',
                       'Medical_History_14', 'Medical_History_17', 'Medical_History_18',
                       'Medical_History_19', 'Medical_History_20', 'Medical_History_22',
                       'Medical_History_23', 'Medical_History_27', 'Medical_History_28',
                       'Medical_History_30', 'Medical_History_31', 'Medical_History_33',
                       'Medical_History_35', 'Medical_History_39', 'Medical_History_40',
                       'Medical_Keyword_3', 'Medical_Keyword_6', 'Medical_Keyword_9',
                       'Medical_Keyword_12', 'Medical_Keyword_15', 'Medical_Keyword_18',
                       'Medical_Keyword_23', 'Medical_Keyword_25', 'Medical_Keyword_28',
                       'Medical_Keyword_29', 'Medical_Keyword_33', 'Medical_Keyword_35',
                       'Medical_Keyword_38', 'Medical_Keyword_41', 'Medical_Keyword_43',
                       'Medical_Keyword_44', 'Medical_Keyword_45', 'Medical_Keyword_47',
                       'Medical_Keyword_48', 'Medical_Keywords_Count']
        
        # Filter and reshape the input data
        selected_data = sample[sample['f'].isin(selected_f)]['v'].values
        input_data = selected_data.reshape((1, -1))
        
        prediction = model.predict(input_data)
        
        # Close and remove the temporary file
        uploaded_file.close()
        os.remove(file_path)
        return render_template("index.html", prediction=prediction)
    return render_template("index.html", error="No file uploaded")

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)

import pickle
import pandas as pd
from flask import Flask, render_template
from waitress  import serve
import os
print("Current working directory:", os.getcwd())
app = Flask(__name__)
model = pickle.load(open('life_Insures_classifier_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    sample = pd.read_csv('sample.csv')
    sample.columns = [ 'f', 'v']
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
    
    #s= sample['v'].values
    #s1= s.reshape((1, 122))
    prediction = model.predict(input_data)
    return render_template("index.html" , prediction=prediction)
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
    

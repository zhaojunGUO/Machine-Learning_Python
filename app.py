import numpy as np
from flask import Flask,request,render_template
import pickle
import pandas as pd

app = Flask(__name__,template_folder="templates")
app.debug=True
model_logistic = pickle.load(open('logistic_regression.sav', 'rb'))
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        features = [x for x in request.form.values()]
        for i in range(len(features)):
            features[i]=float(features[i])
        f=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight','FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC']
        d={}
        for i in range(len(f)):
            d[f[i]]=features[i]
        final_features = pd.DataFrame(d,index=[1])
        result = model_logistic.predict(final_features)
        
        results=['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']
        output = results[result[0]]
        return render_template('Result.html', prediction_text='Your weight level is:-  {}'.format(output))



if __name__=="__main__":
    app.run()
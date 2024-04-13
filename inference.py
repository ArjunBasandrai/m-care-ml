import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def init():
    model = joblib.load('models/model.pkl')

    x_encoders = []
    encoder_dir = "encoders"
    for encoder in os.listdir(encoder_dir):
        path = os.path.join(encoder_dir, encoder)
        le = LabelEncoder()
        le.classes_ = np.load(path, allow_pickle=True)
        x_encoders.append(le)

    sample_data = {
        "Age": "35-40",
        "Sad_Tearful": "Yes",
        "Irritable": "Yes",
        "Trouble_Sleeping": "Yes",
        "Problems_Focusing": "Yes",
        "Eating_Disorder": "Yes",
        "Guilt": "Yes",
        "Problems_Bonding": "No",
        "Suicide_Attempt": "No"
    }

    return model, x_encoders, sample_data

def predict(model, input_data, encoder):
    if list(input_data.keys()) != ['Age', 'Sad_Tearful', 'Irritable', 'Trouble_Sleeping', 'Problems_Focusing', 'Eating_Disorder', 'Guilt', 'Problems_Bonding', 'Suicide_Attempt']:
        print("Error: Invalid Input")
        return None
    
    test_df = pd.DataFrame([input_data])
    for i, column in enumerate(test_df.columns):
        le = encoders[i]
        test_df[column] = le.transform(test_df[column])
    pred = model.predict(test_df)
    return pred[0]

def check(model, input_data, encoders):
    prediction = predict(model, input_data, encoders)
    if prediction == None:
        return None
    return "Signs of depression detected" if prediction else "No signs of depression detected"

model, encoders, data = init()
print(check(model, data, encoders))
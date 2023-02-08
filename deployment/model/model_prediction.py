import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("https://raw.githubusercontent.com/OmdenaAI/Algeria-Chapter-Green/main/Part%202/Subproject3-%20Recommendation%20Systems/data/Irrigation%20Dataset%20-%20NIT%20Raipur.csv", names = ["CropType", "CropDays", "SoilMoisture", "Temperature", "Humidity", "NeedIrrigation"])
X = df[df.columns[:len(df.columns) - 1]]

MODEL = tf.keras.models.load_model("model//ANNmodel.h5")

ct = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), ["CropType"])], remainder = "passthrough")
ct.fit(X)

def predict(param1, param2, param3, param4, param5):
    predictdf = pd.DataFrame({"CropType": [param1],
    "CropDays": [param2],
    "SoilMoisture": [param3],
    "Temperature": [param4], 
    "Humidity": [param5]})
    all_params = ct.transform(predictdf)
    op = MODEL.predict(all_params)
    return int(op[0][0] > 0.5)


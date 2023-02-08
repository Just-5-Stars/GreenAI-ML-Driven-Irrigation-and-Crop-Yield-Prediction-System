import numpy as np
import pandas as pd
import tensorflow as tf
import warnings


df1 = pd.read_csv("https://raw.githubusercontent.com/OmdenaAI/Algeria-Chapter-Green/main/Part%202/Subproject3-%20Recommendation%20Systems/data/Irrigation%20Dataset%20-%20NIT%20Raipur.csv", names = ["CropType", "CropDays", "SoilMoisture", "Temperature", "Humidity", "NeedIrrigation"])
df2 = pd.read_excel("Irrigation Dataset - GCE Kanpur.xlsx")


# Data Preprocessing
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
df2 = df2.drop(['Soil Temperature'], axis=1)
df2.rename(columns = {'Irrigation(Y/N)':'NeedIrrigation', "Soil Moisture": "SoilMoisture", }, inplace = True)
df2['CropType'] = df2['CropType'].replace(1,5)



df = pd.concat([df1, df2])  # Merging the datasets
df.drop(df[(df['Temperature'] > 100)].index, inplace=True)
df = df.sort_values(by=['CropType'])


print("--Preprocessed Data--")
print(df.head())
print(np.unique(df["CropType"]))

# Data split: Features & Labels
features = df[df.columns[:len(df.columns) - 1]]
labels = df[df.columns[-1]]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), ["CropType"])], remainder = "passthrough")
X = np.array(ct.fit_transform(features))
y = np.array(labels)

# Data split: Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# Building the ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 6, activation = "relu"))
model.add(tf.keras.layers.Dense(units = 6, activation = "relu"))
model.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

# Compiling the model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, epochs = 50)

# Testing the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy of the model:", accuracy_score(y_test, y_pred))

# Save the model
model.save("ANNmodel.h5")
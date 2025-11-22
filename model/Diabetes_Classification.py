import os
import pandas as pd
import joblib
######################################################
from  sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
######################################################
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


current_dir = os.path.dirname(os.path.abspath(__file__))
path=os.path.join(current_dir ,"..","Notebooks", "cleaned_diabetes_data.csv")

df=pd.read_csv(path)

X = df.drop(columns='Outcome')
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=y, random_state=42)

ss = StandardScaler()
rf = RandomForestClassifier(class_weight="balanced",bootstrap=True , max_depth=None 
                               , max_features="log2" , min_samples_leaf=1 , min_samples_split=2 
                               , n_estimators=100 ,random_state=42)
model=make_pipeline(ss,rf)

model.fit(X_train, y_train)

joblib.dump(model,os.path.join(current_dir, "diabetes_model.pkl"))

print("Model saved!")


import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# File Paths
model_path = "lgbm_model.sav"
encoding_path = "cat_encods.json"
component_config_path = "component_configs.json"

# predefined
feature_order =['Gender', 'Age', 'Occupation', 'City_Category', 
                'Stay_In_Current_City_Years', 'Marital_Status', 
                'Product_Category_1', 'Product_Category_2']

# Loading the files
model = pickle.load(open(model_path, 'rb'))


# loading the classes & type casting the encoding indexes
classes = json.load(open(encoding_path, "r"))
classes = {k:{int(num):cat for num,cat in v.items() } for k,v in classes.items()}

inverse_class = {col:{val:key for key, val in clss.items()}  for col, clss in classes.items()}

feature_limitations = json.load(open(component_config_path, "r"))


# Code
total_examples = [
    ['M', '18-25', 14.0, 'B', '3', 0.0, 5.0, 6.0, 1735],
    ['M', '36-45', 12.0, 'C', '2', 0.0, 10.0, 13.0, 23628],
    ['F', '36-45', 1.0, 'A', '3', 1.0, 8.0, 9.0, 6094],
    ['M', '26-35', 16.0, 'B', '4+', 1.0, 2.0, 9.0, 12899],
    ['M', '18-25', 15.0, 'C', '3', 0.0, 1.0, 2.0, 15247]
]


# Util function
def decode(col, data):
  return classes[col][data]

def encode(col, str_data):
  return inverse_class[col][str_data]

def feature_decode(df):

  # exclude the target var
  cat_cols = list(classes.keys())
  if "Purchase" in cat_cols:
    cat_cols.remove("Purchase")

  for col in cat_cols:
     df[col] = decode(col, df[col])

  return df

def feature_encode(df):
  
  # exclude the target var
  cat_cols = list(classes.keys())
  if "Purchase" in cat_cols:
    cat_cols.remove("Purchase")
  
  for col in cat_cols:
     df[col] = encode(col, df[col])
  
  return df


def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  features = feature_encode(features)
  features = np.array(features).reshape(-1,len(feature_order))

  # prediction
  pred = model.predict(features) # .predict_proba(features)

  return np.round(pred,5)

# Creating the gui component according to component.json file

def RMSE(actual, pred):
    return np.around(np.sqrt(mean_squared_error([actual],[pred])),3)

inputs = []
with gr.Blocks() as demo:

    for col in feature_order:
      if col in feature_limitations["cat"].keys():
        
        # extracting the params
        vals = feature_limitations["cat"][col]["values"]
        def_val = feature_limitations["cat"][col]["def"]
        
        # creating the component
        inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
      else:
        
        # extracting the params
        min = feature_limitations["num"][col]["min"]
        max = feature_limitations["num"][col]["max"]
        def_val = feature_limitations["num"][col]["def"]
        
        # creating the component
        inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col))
    
    pred_btn = gr.Button("Predict")
    output = gr.Number(label="Prediction")
    pred_btn.click(fn=predict, inputs=inputs, outputs=output) #examples=examples
    
    loss_btn = gr.Button("calculate loss")
    actual = gr.Number(value=11302, label="Actual Value", )
    loss = gr.Number(label="RMSE")
    loss_btn.click(fn=RMSE, inputs=[actual, output], outputs=loss)


    gr.Examples(total_examples, [*inputs, actual])

# Launching the demo
if __name__ == "__main__":
    demo.launch()

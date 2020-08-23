import streamlit as st
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import h2o
from h2o.automl import H2OAutoML
import time
from PIL import Image

#header
image = Image.open('image/header.jpg')
st.image(image,use_column_width=True)

st.title('Data Profiling & Modeling')

## data_load
st.header('Data load')

#uplaod_file
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", encoding="auto")
if uploaded_file is None:
    st.stop()
else:
    data = pd.read_csv(uploaded_file)

# sidebar
option = st.sidebar.selectbox(
    'どの変数を予測したいですか?',
     data.columns)

train_time = st.sidebar.selectbox(
    'どのぐらいの時間を学習させたいですか?',
     (100, 1000, 10000))

## profile
st.header("Profiling")
pr = ProfileReport(data, explorative=True)
st.write(data)
st_profile_report(pr)

## H2O_modeling
st.header("Modeling")
h2o.init()
htrain = h2o.H2OFrame(data)
x = htrain.columns
y = option
x.remove(y)

# visualize
train_aml = H2OAutoML(seed=42, max_runtime_secs=train_time)
train_aml.train(x=x, y=y, training_frame=htrain)
train_lb = train_aml.leaderboard
result = train_lb.head(rows=train_lb.nrows)
result_df = result.as_data_frame(use_pandas = True , header = True)
st.write(result_df)

# feature_importance
st.header("Feature Importance")
model_ids = list(train_lb['model_id'].as_data_frame().iloc[:,0])
gbm_mid = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])
# st.write(gbm_mid.varimp_plot())
st.write(gbm_mid.varimp(use_pandas=True))

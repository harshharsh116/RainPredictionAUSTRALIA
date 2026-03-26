import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import streamlit as st
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.metrics import classification_report

@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS_rainfall_prediction_dataset_cleaned.csv")
    return df
data = load_data()
sample = load_data()
c1 = st.sidebar.checkbox("Show Original data")
if c1:
    st.header("Original Data")
    st.dataframe(data)

data.drop('Date',axis=1,inplace=True)
c2 = st.sidebar.checkbox("EDA")
if c2:
    st.header('EDA')
    st.subheader('top 10 records')
    st.write(data.head(10))

    st.subheader('Last 10 records')
    st.write(data.tail(10))

    st.subheader('Stats of data')
    st.write(data.describe())

    st.subheader('Null Count')
    st.write(data.isnull().sum())

    st.subheader('Correlation Matrix')
    st.write(data.corr(numeric_only=True))

    st.subheader('Columns')
    st.write(data.columns)

    st.subheader('Data Shape')
    st.write(data.shape)


locencodr = LabelEncoder()
windgustdir  = LabelEncoder()
Windir9am = LabelEncoder()
windir3pm = LabelEncoder()
raintoday  = LabelEncoder()
raintomorrow = LabelEncoder()

data['Location'] = locencodr.fit_transform(data['Location'])
data['WindGustDir']= windgustdir.fit_transform(data['WindGustDir'])
data['WindDir9am']= Windir9am.fit_transform(data['WindDir9am'])
data['WindDir3pm']=windir3pm.fit_transform(data['WindDir3pm'])
data['RainToday']=raintoday.fit_transform(data['RainToday'])
data['RainTomorrow']=raintomorrow.fit_transform(data['RainTomorrow'])


x = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

test = st.sidebar.slider('test size', 0.0, 1.0, 0.2, 0.1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test, random_state=42
)

c9 = st.sidebar.checkbox("train test size")
if c9:
    st.subheader("Shape")
    st.write("xtrain",x_train.shape)
    st.write("ytrain",y_train.shape)
    st.write("xtest",x_test.shape)
    st.write("ytest",y_test.shape)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xg = XGBClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8]
 }

grid_search = GridSearchCV(
    estimator=xg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=2,
    n_jobs=1,
    refit=True,
 )
st.sidebar.subheader('To Build Model')
v3 = st.sidebar.checkbox("model build ")
if v3:
 modelname = st.sidebar.selectbox("select model",['linear','xgb','decisontree','randomforest','gradientboost','adaboost','svm','knc','gusianNB','gridsearchmodel','Kerastenserflow'])
 if modelname == 'linear':
    linear = LogisticRegression()
    model = linear.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'xgb':
    xgb = XGBClassifier()
    model = xgb.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'decisontree':
    decissiontree = DecisionTreeClassifier()
    model =decissiontree.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'randomforest':
    randomforest = RandomForestClassifier()
    model = randomforest.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'gradientboost':
    gradientbooster = GradientBoostingClassifier()
    model=gradientbooster.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'adaboost':
    adaboost = AdaBoostClassifier()
    model=adaboost.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'svm':
    svm = SVC(probability=True)
    model=svm.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'knc':
    knc = KNeighborsClassifier()
    model=knc.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'gusianNB':
    gusianNB = GaussianNB()
    model=gusianNB.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'gridsearchmodel':
    model=grid_search.fit(x_train, y_train)
    scores = cross_val_score(model, x_train, y_train, cv=5)
    st.write("Scores Cross validation:", scores)
    st.write("scores mean", np.mean(scores))
 elif modelname == 'Kerastenserflow':

     from tensorflow import keras

     model = keras.Sequential([
         keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
         keras.layers.Dense(32, activation='relu'),
         keras.layers.Dense(1, activation='sigmoid')
     ])

     model.compile(
         optimizer='adam',
         loss='binary_crossentropy',
         metrics=['accuracy']
     )

     model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
     loss, accuracy = model.evaluate(x_test, y_test)
     st.write("Test data Accuracy",accuracy)



 def result(model, x_test, y_test):
     ypred = model.predict(x_test)
     ans = classification_report(y_test, ypred,output_dict=True)
     df_report = pd.DataFrame(ans).transpose()
     return df_report,ypred



 b2 = st.sidebar.button("view result")

 if b2:
     if modelname != 'Kerastenserflow':
         accuracy,ypred = result(model, x_test, y_test)
         st.text("Result:")
         st.text(accuracy)

     else:
         ypred = (model.predict(x_test) > 0.5).astype(int)
         accuracy = classification_report(y_test, ypred,output_dict=True)
         ans = pd.DataFrame(accuracy).transpose()
         st.text("Accuracy:")
         st.write(ans)

     cm = confusion_matrix(y_test, ypred)

     fig, ax = plt.subplots()
     sns.heatmap(cm, annot=True, fmt='d', ax=ax)
     st.pyplot(fig)

     scores = cross_val_score(model, x_train, y_train, cv=10)
     st.write("Scores:",scores)
     st.write("scores mean",np.mean(scores))

st.title("RainPrediction For Tomorrow")
col1,col2 = st.columns([1,1])
pred = st.sidebar.checkbox("prediction")
if pred:
    if not v3:
        st.error("First Build Model")
    else:

        Location = col1.selectbox("Select Location",sample['Location'].unique())
        MinTemp = col2.number_input("Select Min Temp",-11.0,40.0)
        MaxTemp = col1.number_input("Select Max Temp",-11.0,40.0)
        Rainfall = col2.number_input("Select Rainfall(mm)",sample['Rainfall'].min(),sample['Rainfall'].max())
        Evaporation = col1.number_input("Evaporation(mm)",sample['Evaporation'].min(),sample['Evaporation'].max())
        Sunshine = col2.number_input("Sunshine(hours a day)",sample['Sunshine'].min(),sample['Sunshine'].max())
        WindGustDir = col1.selectbox("Wind direction for strongest wind of a day",sample['WindGustDir'].unique())
        WindGustSpeed = col2.number_input("Speed of Strongest wind of a day",sample['WindGustSpeed'].min(),sample['WindGustSpeed'].max())
        WindDir9am = col1.selectbox("Wind direction for 9am",sample['WindDir9am'].unique())
        WindDir3pm = col2.selectbox("Wind direction for 3PM",sample['WindDir3pm'].unique())
        WindSpeed9am = col1.number_input("Wind speed 9am",sample['WindSpeed9am'].min(),sample['WindSpeed9am'].max())
        WindSpeed3pm = col2.number_input("Wind speed 3pm",sample['WindSpeed3pm'].min(),sample['WindSpeed3pm'].max())
        Humidity9am = col1.number_input("Humidity9am(%)",sample['Humidity9am'].min(),sample['Humidity9am'].max())
        Humidity3pm = col2.number_input("Humidity3pm(%)",sample['Humidity3pm'].min(),sample['Humidity3pm'].max())
        Pressure9am = col1.number_input("Pressure9am",sample['Pressure9am'].min(),sample['Pressure9am'].max())
        Pressure3pm = col2.number_input("Pressure3pm",sample['Pressure3pm'].min(),sample['Pressure3pm'].max())
        Cloud9am = col1.number_input("Cloud9am(fraction of sky covered by clouds)",sample['Cloud9am'].min(),sample['Cloud9am'].max())
        Cloud3pm = col2.number_input("Cloud3pm(fraction of sky covered by clouds)",sample['Cloud3pm'].min(),sample['Cloud3pm'].max())
        Temp9am = col1.number_input("Temp9am",sample['Temp9am'].min(),sample['Temp9am'].max())
        Temp3pm = col2.number_input("Temp3pm",sample['Temp3pm'].min(),sample['Temp3pm'].max())
        RainToday = col1.selectbox("RainToday",sample['RainToday'].unique())


        #check = pd.DataFrame([{
        #    'Location' : [Location],
        #    'MinTemp' : [MinTemp],
        #    'MaxTemp' : [MaxTemp],
        #    'Rainfall' : [Rainfall],
        #    'Evaporation' : [Evaporation],
        #    'Sunshine' : [Sunshine],
        #    'WindGustDir' : [WindGustDir],
        #    'WindGustSpeed' : [WindGustSpeed],
        #    'WindDir9am' : [WindDir9am],
        #    'WindDir3pm' : [WindDir3pm],
        #    'WindSpeed9am' : [WindSpeed9am],
        #    'WindSpeed3pm' : [WindSpeed3pm],
        #    'Humidity9am' : [Humidity9am],
        #    'Humidity3pm' : [Humidity3pm],
        #    'Pressure9am' : [Pressure9am],
        #    'Pressure3pm' : [Pressure3pm],
        #    'Cloud9am' : [Cloud9am],
        #    'Cloud3pm' : [Cloud3pm],
        #    'Temp9am' : [Temp9am],
        #    'Temp3pm' : [Temp3pm],
        #    'RainToday' : [RainToday]
        #}])

        check = pd.DataFrame([{
            'Location': locencodr.transform([Location])[0],
            'MinTemp': MinTemp,
            'MaxTemp': MaxTemp,
            'Rainfall': Rainfall,
            'Evaporation': Evaporation,
            'Sunshine': Sunshine,
            'WindGustDir': windgustdir.transform([WindGustDir])[0],
            'WindGustSpeed': WindGustSpeed,
            'WindDir9am': Windir9am.transform([WindDir9am])[0],
            'WindDir3pm': windir3pm.transform([WindDir3pm])[0],
            'WindSpeed9am': WindSpeed9am,
            'WindSpeed3pm': WindSpeed3pm,
            'Humidity9am': Humidity9am,
            'Humidity3pm': Humidity3pm,
            'Pressure9am': Pressure9am,
            'Pressure3pm': Pressure3pm,
            'Cloud9am': Cloud9am,
            'Cloud3pm': Cloud3pm,
            'Temp9am': Temp9am,
            'Temp3pm': Temp3pm,
            'RainToday': raintoday.transform([RainToday])[0]
        }])

        check = scaler.transform(check)

        if st.button("Predict Rain Tomorrow"):

            if modelname == 'Kerastenserflow':
                pred = (model.predict(check) > 0.5).astype(int)[0][0]
            else:
                pred = model.predict(check)[0]

            if pred == 1:
                st.success("🌧️ Rain Tomorrow: YES")
            else:
                st.success("☀️ Rain Tomorrow: NO")


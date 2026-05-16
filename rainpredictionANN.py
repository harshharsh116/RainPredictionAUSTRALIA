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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("RainPrediction For Tomorrow")

@st.cache_data
def load_data():
    return pd.read_csv("weatherAUS_rainfall_prediction_dataset_cleaned.csv")

data = load_data()
sample_data = data.copy()

c1 = st.sidebar.checkbox("Show Original data")

if c1:
    st.header("Original Data")
    st.dataframe(data.head(30))

data.drop('Date', axis=1, inplace=True)

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
windgustdir = LabelEncoder()
Windir9am = LabelEncoder()
windir3pm = LabelEncoder()
raintoday = LabelEncoder()
raintomorrow = LabelEncoder()

data['Location'] = locencodr.fit_transform(data['Location'])
data['WindGustDir'] = windgustdir.fit_transform(data['WindGustDir'])
data['WindDir9am'] = Windir9am.fit_transform(data['WindDir9am'])
data['WindDir3pm'] = windir3pm.fit_transform(data['WindDir3pm'])
data['RainToday'] = raintoday.fit_transform(data['RainToday'])
data['RainTomorrow'] = raintomorrow.fit_transform(data['RainTomorrow'])

x = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

test = st.sidebar.slider('test size', 0.1, 0.5, 0.2, 0.1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test, random_state=42
)

c9 = st.sidebar.checkbox("train test size")

if c9:
    st.subheader("Shape")
    st.write("xtrain", x_train.shape)
    st.write("ytrain", y_train.shape)
    st.write("xtest", x_test.shape)
    st.write("ytest", y_test.shape)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xg = XGBClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6]
}

grid_search = GridSearchCV(
    estimator=xg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=2,
    n_jobs=1,
    refit=True,
)

if "model" not in st.session_state:
    st.session_state.model = None

if "modelname" not in st.session_state:
    st.session_state.modelname = None

st.sidebar.subheader('To Build Model')

modelname = st.sidebar.selectbox(
    "select model",
    [
        'linear',
        'xgb',
        'decisontree',
        'randomforest',
        'gradientboost',
        'adaboost',
        'svm',
        'knc',
        'gusianNB',
        'gridsearchmodel',
        'Kerastenserflow'
    ]
)

build = st.sidebar.button("Build Model")

if build:

    if modelname == 'linear':
        model = LogisticRegression()

    elif modelname == 'xgb':
        model = XGBClassifier()

    elif modelname == 'decisontree':
        model = DecisionTreeClassifier()

    elif modelname == 'randomforest':
        model = RandomForestClassifier()

    elif modelname == 'gradientboost':
        model = GradientBoostingClassifier()

    elif modelname == 'adaboost':
        model = AdaBoostClassifier()

    elif modelname == 'svm':
        model = SVC(probability=True)

    elif modelname == 'knc':
        model = KNeighborsClassifier()

    elif modelname == 'gusianNB':
        model = GaussianNB()

    elif modelname == 'gridsearchmodel':
        model = grid_search

    elif modelname == 'Kerastenserflow':

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

        model.fit(
            x_train,
            y_train,
            epochs=10,
            batch_size=32,
            verbose=0
        )

        loss, accuracy = model.evaluate(x_test, y_test)

        st.write("Test data Accuracy", accuracy)

    if modelname != 'Kerastenserflow':

        model.fit(x_train, y_train)

        scores = cross_val_score(model, x_train, y_train, cv=5)

        st.write("Scores Cross validation:", scores)
        st.write("scores mean", np.mean(scores))

    st.session_state.model = model
    st.session_state.modelname = modelname

    st.success("Model Built Successfully")

def result(model, x_test, y_test):

    ypred = model.predict(x_test)

    ans = classification_report(
        y_test,
        ypred,
        output_dict=True
    )

    df_report = pd.DataFrame(ans).transpose()

    return df_report, ypred

b2 = st.sidebar.button("view result")

if b2:

    if st.session_state.model is None:

        st.error("First Build Model")

    else:

        model = st.session_state.model
        modelname = st.session_state.modelname

        if modelname != 'Kerastenserflow':

            accuracy, ypred = result(model, x_test, y_test)

            st.text("Result:")
            st.write(accuracy)

        else:

            ypred = (model.predict(x_test) > 0.5).astype(int)

            accuracy = classification_report(
                y_test,
                ypred,
                output_dict=True
            )

            ans = pd.DataFrame(accuracy).transpose()

            st.text("Accuracy:")
            st.write(ans)

        cm = confusion_matrix(y_test, ypred)

        fig, ax = plt.subplots()

        sns.heatmap(cm, annot=True, fmt='d', ax=ax)

        st.pyplot(fig)

col1, col2 = st.columns([1, 1])

pred = st.sidebar.checkbox("prediction")

if pred:

    if st.session_state.model is None:

        st.error("First Build Model")

    else:

        model = st.session_state.model
        modelname = st.session_state.modelname

        Location = col1.selectbox(
            "Select Location",
            sample_data['Location'].unique()
        )

        MinTemp = col2.number_input(
            "Select Min Temp",
            -11.0,
            40.0
        )

        MaxTemp = col1.number_input(
            "Select Max Temp",
            -11.0,
            40.0
        )

        Rainfall = col2.number_input(
            "Select Rainfall(mm)",
            float(sample_data['Rainfall'].min()),
            float(sample_data['Rainfall'].max())
        )

        Evaporation = col1.number_input(
            "Evaporation(mm)",
            float(sample_data['Evaporation'].min()),
            float(sample_data['Evaporation'].max())
        )

        Sunshine = col2.number_input(
            "Sunshine(hours a day)",
            float(sample_data['Sunshine'].min()),
            float(sample_data['Sunshine'].max())
        )

        WindGustDir = col1.selectbox(
            "Wind direction for strongest wind of a day",
            sample_data['WindGustDir'].unique()
        )

        WindGustSpeed = col2.number_input(
            "Speed of Strongest wind of a day",
            float(sample_data['WindGustSpeed'].min()),
            float(sample_data['WindGustSpeed'].max())
        )

        WindDir9am = col1.selectbox(
            "Wind direction for 9am",
            sample_data['WindDir9am'].unique()
        )

        WindDir3pm = col2.selectbox(
            "Wind direction for 3PM",
            sample_data['WindDir3pm'].unique()
        )

        WindSpeed9am = col1.number_input(
            "Wind speed 9am",
            float(sample_data['WindSpeed9am'].min()),
            float(sample_data['WindSpeed9am'].max())
        )

        WindSpeed3pm = col2.number_input(
            "Wind speed 3pm",
            float(sample_data['WindSpeed3pm'].min()),
            float(sample_data['WindSpeed3pm'].max())
        )

        Humidity9am = col1.number_input(
            "Humidity9am(%)",
            float(sample_data['Humidity9am'].min()),
            float(sample_data['Humidity9am'].max())
        )

        Humidity3pm = col2.number_input(
            "Humidity3pm(%)",
            float(sample_data['Humidity3pm'].min()),
            float(sample_data['Humidity3pm'].max())
        )

        Pressure9am = col1.number_input(
            "Pressure9am",
            float(sample_data['Pressure9am'].min()),
            float(sample_data['Pressure9am'].max())
        )

        Pressure3pm = col2.number_input(
            "Pressure3pm",
            float(sample_data['Pressure3pm'].min()),
            float(sample_data['Pressure3pm'].max())
        )

        Cloud9am = col1.number_input(
            "Cloud9am(fraction of sky covered by clouds)",
            float(sample_data['Cloud9am'].min()),
            float(sample_data['Cloud9am'].max())
        )

        Cloud3pm = col2.number_input(
            "Cloud3pm(fraction of sky covered by clouds)",
            float(sample_data['Cloud3pm'].min()),
            float(sample_data['Cloud3pm'].max())
        )

        Temp9am = col1.number_input(
            "Temp9am",
            float(sample_data['Temp9am'].min()),
            float(sample_data['Temp9am'].max())
        )

        Temp3pm = col2.number_input(
            "Temp3pm",
            float(sample_data['Temp3pm'].min()),
            float(sample_data['Temp3pm'].max())
        )

        RainToday = col1.selectbox(
            "RainToday",
            sample_data['RainToday'].unique()
        )

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

                pred = (
                    model.predict(check) > 0.5
                ).astype(int)[0][0]

            else:

                pred = model.predict(check)[0]

            if pred == 1:
                st.success("🌧️ Rain Tomorrow: YES")
            else:
                st.success("☀️ Rain Tomorrow: NO")

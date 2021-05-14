# %matplotlib notebook
import os
from sklearn.svm import SVC
from Metadata import getmetadata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

import pandas as pd
import numpy as np

app = Flask(__name__)


def predict_genre(file):
    # ML Code
    df = pd.read_csv('data.csv')
    df = df.drop(['beats'], axis=1)

    df['class_name'] = df['class_name'].astype('category')
    df['class_label'] = df['class_name'].cat.codes

    lookup_genre_name = dict(zip(df.class_label.unique(), df.class_name.unique()))

    cols = list(df.columns)
    cols.remove('label')
    cols.remove('class_label')
    cols.remove('class_name')
    df[cols]

    X = df.iloc[:, 1:28]
    y = df['class_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)


    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)


    # %matplotlib notebook
    clf = RandomForestClassifier(
        random_state=0, n_jobs=-1).fit(X_train_scaled, y_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns.values[i] for i in indices]


    clf = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, y_train)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns.values[i] for i in indices]

    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(X_train_scaled, y_train)
    knn.score(X_test_scaled, y_test)

    a = getmetadata(file)
    d1 = np.array(a)
    data1 = scaler.transform([d1])

    # b = getmetadata("hiphop1.wav")
    # d2 = np.array(b)
    # data2 = scaler.transform([d2])

    genre_prediction = knn.predict(data1)
    print(lookup_genre_name[genre_prediction[0]])

    # genre_prediction = knn.predict(data2)
    # print(lookup_genre_name[genre_prediction[0]])

    clf = SVC(kernel='linear', C=10).fit(X_train_scaled, y_train)
    clf.score(X_test_scaled, y_test)

    genre_prediction = clf.predict(data1)
    print(lookup_genre_name[genre_prediction[0]])

    return lookup_genre_name[genre_prediction[0]]

    # genre_prediction = clf.predict(data2)
    # print(lookup_genre_name[genre_prediction[0]])


# app.config["uploads"] = "C:\Users\DELL\Desktop\6\Music_Genere\uploads"

@app.route('/',  methods=['POST', 'GET'])
def index():
    if request.method == "GET":
        return render_template('index.html', genre="")

    elif request.method == "POST":
        
        file = request.files['audio']
         
        print(file)
        genre = predict_genre(file)
        print(genre)

        return render_template('index.html', genre=genre)

if __name__ == '__main__':
    app.run(debug=True)

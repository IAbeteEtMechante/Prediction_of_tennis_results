import streamlit as st 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from PIL import Image
# import matplotlib.pyplot as plt

import numpy as np
st.write("""
## Who will win this game?
""")

interview = st.text_area("Interview content:", "I am going to win because I am the best")

filename = st.sidebar.file_uploader('Upload the interview here:')

player = st.sidebar.text_input("Player", "Federer")

opponent = st.sidebar.text_input("Opponent", "Nadal")

dataset_type = st.sidebar.selectbox("Select Analysis Type:",("-","Text Analysis","Numerical Analysis", "Text and Numerical Analysis"))

# dataset_type = st.sidebar.selectbox("Select Type:",("Not Pierre","Text Analysis", "Numerical Analysis", "Text and Numerical Analysis","Classification","Regression"))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y= data.target

    return X,y

def get_type(dataset_type):
    if dataset_type == "Classification":
        dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer", "Wine dataset"))
        classfier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest"))
    else:
        dataset_name = st.sidebar.selectbox("Select Dataset", ("Diabetes","Breast Cancer", "Wine dataset"))
        classfier_name = st.sidebar.selectbox("Select Classifier", ("Regression","SVM","Random Forest"))

    return dataset_name, classfier_name
    
dataset_name, classfier_name = get_type(dataset_type)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth",2,15)
        n_estimaor = st.sidebar.slider("Number of Estimators",2,100)
        params["max_depth"] = max_depth
        params["n_estimaor"] = n_estimaor
    return params

params = add_parameter_ui(classfier_name)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimaor"],
                                    max_depth= params["max_depth"],random_state=1234)
    return clf

clf = get_classifier(classfier_name,params)

def run_text_analysis():
    df = pd.read_csv("../data/interviews/interviews_en.csv")
    df_class = pd.read_csv('../data/target.csv', index_col=0)
    df['Class'] = df_class['Class']
    # split X and y into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.Class, random_state=42)
    # st.write("X_train.shape : ", X_train.shape)
    # st.write("X_test.shape : ", X_test.shape)
    vect = CountVectorizer(min_df =1, stop_words='english' ,token_pattern=r'\b[a-zA-Z]+\b')
    X_train_dtm = vect.fit_transform(X_train)
    logReg = LogisticRegression(solver='liblinear')
    logReg.fit(X_train_dtm,y_train)

    #predict winner on our interview:
    X_sample = [interview]
    X_sample_dtm = vect.transform(X_sample)
    result = logReg.predict(X_sample_dtm)
    # result = [0]
    if result[0] == 1:
        st.write("The winner will be: ", player)
        image_file = './Images/' + player.lower()+'.jpeg'
        image_object = Image.open(image_file)
        st.image(image_object, caption='WINNEEEEEERR')
    else:
        st.write("The winner will be: ", opponent)
        image_file = './Images/' + opponent.lower()+'.jpeg'
        image_object = Image.open(image_file)
        st.image(image_object, caption='WINNEEEEEERR')        

    #confidence level
    X_test_dtm = vect.transform(X_test)
    y_pred_class = logReg.predict(X_test_dtm)
    metrics.accuracy_score(y_test, y_pred_class)
    st.write("Confidence Level: ", metrics.accuracy_score(y_test, y_pred_class))



#classification

if dataset_type == "Text Analysis":
    run_text_analysis()



# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state =1234)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test,y_pred)
# confMatrix = confusion_matrix(y_test,y_pred)
# st.write(f"classfier = {classfier_name}")
# st.write(f"accuracy = {acc}")
# st.write(f"Confustion Matrix: {confMatrix}")

#Plot 
# pca = PCA(2)
# X_projected = pca.fit_transform(X)
# x1 = X_projected[:,0]
# x2 = X_projected[:,1]
# fig = plt.figure()
# plt.scatter(x1,x2, c=y, alpha =0.8, cmap ="viridis")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.colorbar()

# st.pyplot()
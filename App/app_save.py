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
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

import numpy as np
st.write("""
## Who will win this game?
""")

interview = st.text_area("Interview content:", "I am going to win because I am the best")

# filename = st.sidebar.file_uploader('Upload the interview here:')

filename_numerical_predict = st.sidebar.file_uploader('Upload numerical data of this match here:')

player = st.sidebar.text_input("Player", "Federer")

opponent = st.sidebar.text_input("Opponent", "Nadal")

dataset_type = st.sidebar.selectbox("Select Analysis Type:",("-","Text Analysis","Magical Text Analysis","Numerical Analysis", "Text and Numerical Analysis"))

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

def numerical_transformation(df):
    from sklearn import preprocessing
    df.index = list(range(0, df.shape[0]))
    df.drop(['Match', 'ID', 'Result', 'Interviewee', 'Opponent'], axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"] )
    column_date = df['Date']

    df_date = pd.DataFrame({"year": column_date.dt.year,
                            "dayofyear": column_date.dt.dayofyear,
                            "week": column_date.dt.week,
                            "dayofweek": column_date.dt.dayofweek,
                           })
    df.drop('Date', axis=1, inplace=True)
    df = pd.concat([df, df_date], axis=1)
    df['is_male'] = df['Sex'].map(lambda s: 1 if s == 'Male' else 0)
    df.drop('Sex', axis=1, inplace=True)
    df['Rank_T'] = 0
    for index, row in df.iterrows():
        if row['Sport'] != 'Tennis':
            df.loc[(df.index==index), 'Rank_T'] = rank_transform(row['Rank'])
        else:
            df.loc[(df.index==index), 'Rank_T'] = float(row['Rank'])
    df['Rank'] = df['Rank_T']
    df['Rank_T'] = 0
    for index, row in df.iterrows():
        if row['Sport'] != 'Tennis':
            df.loc[(df.index==index), 'Rank_T'] = rank_transform(row['Rank Opp.'])
        else:
            df.loc[(df.index==index), 'Rank_T'] = float(row['Rank Opp.'])
    df['Rank Opp.'] = df['Rank_T']
    df.drop('Rank_T', axis=1, inplace=True)
    df['Final Result'] = df['Final Result'].apply(lambda x: 1 if x=='YES' else 0)
    df['is_int_prime_age'] = df['Int. Age'].apply(lambda x: 1 if 23 <= x <= 28 else 0)
    df['is_opp_prime_age'] = df['Opp. Age'].apply(lambda x: 1 if 23 <= x <= 28 else 0)
    df['age_gap'] = df['Int. Age'] - df['Opp. Age']

    # Normalization for numerical features
    scaled_cols = df[['Rank','Rank Opp.', 'Int. Age', 'Opp. Age', 'age_gap']]
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_cols = min_max_scaler.fit_transform(scaled_cols)
    df[['Rank','Rank Opp.', 'Int. Age', 'Opp. Age', 'age_gap']] = scaled_cols

    # Create dummy variables for categorical features
    df[['Health', 'Psychics', 'Prev. Match', 'Confidence','year', 'dayofyear', 'week', 'dayofweek','is_male', 'is_int_prime_age', 'is_opp_prime_age']] = df[['Health', 'Psychics', 'Prev. Match', 'Confidence','year', 'dayofyear', 'week', 'dayofweek', 'is_male','is_int_prime_age', 'is_opp_prime_age']].astype('category')
    df = pd.get_dummies(df, drop_first=True)

    df = df.rename(columns={'Final Result': 'Class'})
    df_template = pd.DataFrame(columns=list(df.columns))
    return df, df_template

# rank transform for numerical data
def rank_transform(v):
    v = v.replace('-', ' ')
    split = v.split(' ')
    wins = float(split[0])
    loses = float(split[1])
    loses = loses if loses > 0 else 0.1
    rank = wins / loses
    return rank

def get_data(dataset_type):
    if dataset_type == "Text Analysis":
        df = pd.read_csv("../data/interviews/interviews_en.csv")
        df_class = pd.read_csv('../data/target.csv', index_col=0)
        df['Class'] = df_class['Class']
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.text, df.Class, random_state=42)
        # st.write("X_train.shape : ", X_train.shape)
        # st.write("X_test.shape : ", X_test.shape)
        vect = CountVectorizer(min_df =1, stop_words='english' ,token_pattern=r'\b[a-zA-Z]+\b')
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        X_interview = [interview]
        X_interview_dtm = vect.transform(X_interview)
        #renaming
        X_train, X_test, X_interview = X_train_dtm, X_test_dtm, X_interview_dtm

    elif dataset_type == "Magical Text Analysis":
        df = pd.read_csv("../data/interviews/interviews_en.csv")
        #stem
        df = df[['label','text']]
        df.head(1)

        df_class = pd.read_csv('../data/target.csv', index_col=0)
        df['Class'] = df_class['Class']
        df.loc[50,:]= df.loc[49,:]
        df.loc[50,'text'] = interview

        import re, string, unicodedata
        import nltk
        # import contractions
        # import inflect
        # from bs4 import BeautifulSoup
        from nltk import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import LancasterStemmer, WordNetLemmatizer
        from stop_words import get_stop_words
        from string import ascii_letters, digits, whitespace

        import glob
        import errno

        def tokenize(text):
            words = nltk.word_tokenize(text)
            return words
        def is_ascii(word):
            for c in word:
                if c in ascii_letters:
                    return True
            return False 
        def to_lowercase(words):
            """Convert all characters to lowercase from list of tokenized words"""
            new_words = []
            for word in words:
                new_word = word.lower()
                new_words.append(new_word)
            return new_words   
        def remove_punctuation(words):
            """Remove punctuation from list of tokenized words"""
            new_words = []
            for word in words:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
            return new_words

        def replace_numbers(words):
            """Replace all interger occurrences in list of tokenized words with textual representation"""
            p = inflect.engine()
            new_words = []
            for word in words:
                if word.isdigit():
                    new_word = 'число_' + str(word)
                    new_words.append(new_word)
                else:
                    new_words.append(word)
            return new_words        

        def remove_numbers(words):
            """Remove all interger occurrences in list of tokenized words"""
            new_words = []
            for word in words:
                if not word.isdigit():
                    new_words.append(word)
            return new_words 
        
        def remove_stopwords(words):
            """Remove stop words from list of tokenized words"""
            new_words = []
            for word in words:
                if word not in get_stop_words('bg'):
                    new_words.append(word)
            return new_words   
        def remove_empty_words(words):
            new_words = []
            for word in words:
                if word.strip():
                    new_words.append(word)
            return new_words  

        def print_words(df):
            for i, words in enumerate(df['words'], 1):
                print('Interview ' + str(i))
                print(words)     
        df['words'] = [tokenize(text) for text in df['text']]
        #print_words(df)

        df['words'] = [to_lowercase(words) for words in df['words']]
        # print_words(df)

        df['words'] = [remove_punctuation(words) for words in df['words']]
        # print_words(df)

        df['words'] = [remove_numbers(words) for words in df['words']]
        # print_words(df)

        df['words'] = [remove_stopwords(words) for words in df['words']]
        # print_words(df)

        df['words'] = [remove_empty_words(words) for words in df['words']]

        from nltk.stem import PorterStemmer
        from nltk.stem import LancasterStemmer
        from nltk.tokenize import word_tokenize 
           
        ps = PorterStemmer()
        ls = LancasterStemmer()
        # print_words(df)

        df_stem = pd.DataFrame()
        df_stem['words_stem_1'] = df.words.apply(lambda x : [ps.stem(word) for word in x])
        df_stem['words_stem_2'] = df.words.apply(lambda x : [ls.stem(word) for word in x])

        from gensim.corpora import Dictionary
        from gensim.models import NormModel
        from gensim.models import TfidfModel
        def tf_idf(df, attr):
            documents = df[attr]
            dictionary = Dictionary(documents)
            n_items = len(dictionary)
            #docbow converts to bag of words
            corpus = [dictionary.doc2bow(text) for text in documents]
            #then we apply tfidf 
            tfidf = TfidfModel(corpus) #fit tfidf on this corpus
            corpus_tfidf = tfidf[corpus] #transform the corpus
            
            #then make a dataframe out of it
            ds = []
            for doc in corpus_tfidf:
                d = [0] * n_items
                for index, value in doc :
                    d[index]  = value
                ds.append(d)
            df_tfidf = pd.DataFrame(ds)
            return df_tfidf   

        #we apply the tfidf on each stemmer
        df_tfidf_1 = tf_idf(df_stem, 'words_stem_1')
        df_tfidf_2 = tf_idf(df_stem, 'words_stem_2')      

        def get_headers(df, attr):
            documents = df[attr]
            dictionary = Dictionary(documents)
            return list(dictionary.values())


        df_tfidf_headers_1 = get_headers(df_stem, 'words_stem_1')
        df_tfidf_headers_2 = get_headers(df_stem, 'words_stem_2')

        df_tfidf_1.columns = df_tfidf_headers_1
        df_tfidf_2.columns = df_tfidf_headers_2


        df =df_tfidf_1.copy()

        # df_class = pd.read_csv('../data/target.csv', index_col=0)
        # df['Class'] = df_class['Class']
        # df.loc[50,:]= df.loc[49,:]
        # df.loc[50,'text'] = interview

        top_1_features = ['young','ye','won', 'whatev', 'us', 'unit', 'two', 'twice', 'tough', 'strong', 'stand', 'speed', 'sofia', 'shape', 'second', 'same', 'respect', 'real', 'qualiti', 'put', 'pulev', 'prove', 'pressur', 'press', 'partner', 'over', 'out', 'otherwis', 'or', 'noth', 'need', 'motiv', 'mani', 'lot', 'look', 'kubrat', 'knock', 'keep', 'judg', 'hit', 'here', 'healthi', 'game', 'friend', 'fan', 'fact', 'everyth', 'due', 'drive', 'cours', 'coach', 'clay', 'children', 'chain', 'bulgarian', 'break', 'both', 'big', 'between', 'almost', 'achiev']
        #code blah
        df_top_1 = df.loc[:, top_1_features].copy()

        print(type(df_top_1.loc[50,:]))
        X_interview = df_top_1.loc[50,:].to_frame().T

        print("X_interview shape",X_interview.shape)


        df = df_top_1.loc[:49,:].copy()
        print("df shape",df.shape)
        df_class = pd.read_csv('../data/target.csv', index_col=0)
        # df['Class'] = df_class['Class']


        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df, df_class, random_state=42)
        print("X_train_shape",X_train.shape)
        # st.write("X_train.shape : ", X_train.shape)
        # st.write("X_test.shape : ", X_test.shape)
        # vect = CountVectorizer(min_df =1, stop_words='english' ,token_pattern=r'\b[a-zA-Z]+\b')
        # X_train_dtm = vect.fit_transform(X_train)
        # X_test_dtm = vect.transform(X_test)
        # X_interview = [interview]
        # X_interview_dtm = vect.transform(X_interview)
        # #renaming
        # X_train, X_test, X_interview = X_train_dtm,X_test_dtm,X_interview_dtm

        # if dataset_type == "Text Analysis":
        #         df = pd.read_csv("../data/interviews/interviews_en.csv")
        #         df_class = pd.read_csv('../data/target.csv', index_col=0)
        #         df['Class'] = df_class['Class']
        #         # split X and y into training and testing sets
        #         X_train, X_test, y_train, y_test = train_test_split(df.text, df.Class, random_state=42)
        #         # st.write("X_train.shape : ", X_train.shape)
        #         # st.write("X_test.shape : ", X_test.shape)
        #         vect = CountVectorizer(min_df =1, stop_words='english' ,token_pattern=r'\b[a-zA-Z]+\b')
        #         X_train_dtm = vect.fit_transform(X_train)
        #         X_test_dtm = vect.transform(X_test)
        #         X_interview = [interview]
        #         X_interview_dtm = vect.transform(X_interview)
        #         #renaming
        #         X_train, X_test, X_interview = X_train_dtm, X_test_dtm, X_interview_dtm
        
    elif dataset_type == "Numerical Analysis":
        df = pd.read_csv("../data/Features_Dataset.csv")
        df_transformed, df_template = numerical_transformation(df)
        # X_interview = pd.read_csv(filename_numerical_predict)
        X_interview = pd.read_csv("../data/numerical_input_template.csv")
        X_interview_transformed = numerical_transformation(X_interview)
        X_interview_in_template = df_template.append(X_interview_transformed)
        X_interview = X_interview_in_template.fillna(0).drop('Class', axis=1)
        
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop('Class', axis=1),
                                                            df_transformed.Class, random_state=42)

    else:

        df = pd.read_csv("../data/interviews/interviews_en.csv")
        df_class = pd.read_csv('../data/target.csv', index_col=0)
        df['Class'] = df_class['Class']
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.text, df.Class, random_state=42)
        # st.write("X_train.shape : ", X_train.shape)
        # st.write("X_test.shape : ", X_test.shape)
        vect = CountVectorizer(min_df =1, stop_words='english' ,token_pattern=r'\b[a-zA-Z]+\b')
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)
        X_interview = [interview]
        X_interview_dtm = vect.transform(X_interview)
        #renaming
        X_train, X_test, X_interview = X_train_dtm,X_test_dtm,X_interview_dtm

    return X_train, X_test, X_interview, y_train, y_test
    
X_train, X_test, X_interview, y_train, y_test = get_data(dataset_type)
#for debugging:
# st.write(X_train.shape, X_test.shape, X_interview.shape, y_train.shape, y_test.shape)


classfier_name = st.sidebar.selectbox("Select Classifier", ("Regression","Random Forest"))

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

#classification

if dataset_type != "-":

    # logReg = LogisticRegression(solver='liblinear')
    clf.fit(X_train,y_train)

    #predict winner on our interview:
    result = clf.predict(X_interview)
    # result = [0]
    if result[0] == 1:
        st.write("The winner will be: ", player)
        image_file = './Images/' + player.lower()+'.jpeg'
        image_object = Image.open(image_file)
        st.image(image_object, caption='WINNEEEEEERR')
        y_pred_prob = clf.predict_proba(X_interview)[:, 1]
        st.write("Probability of winning", y_pred_prob[0])
    else:
        st.write("The winner will be: ", opponent)
        image_file = './Images/' + opponent.lower()+'.jpeg'
        image_object = Image.open(image_file)
        st.image(image_object, caption='WINNEEEEEERR')     
        y_pred_prob = clf.predict_proba(X_interview)[:, 1]
        st.write("Probability of winning", 1-y_pred_prob[0])  

    #confidence level


    y_pred_class = clf.predict(X_test)
    metrics.accuracy_score(y_test, y_pred_class)
    st.write("Model accuracy: ", metrics.accuracy_score(y_test, y_pred_class))




    



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
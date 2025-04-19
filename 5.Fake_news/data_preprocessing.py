import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    df = pd.read_csv('train.csv')
    df = df.fillna('')
    X = df['text']  # News content
    y = df['label'] # 0 = REAL, 1 = FAKE
    return X, y

def preprocess_data(X, y):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer

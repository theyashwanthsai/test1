
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
import random
import json
from keras.models import load_model
import numpy as np
import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
from flask import Flask, render_template, request, redirect, url_for, session
import pyrebase
import os
from dotenv import load_dotenv
from functools import wraps

nltk.download('popular')
lemmatizer = WordNetLemmatizer()
load_dotenv()


chat_model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))



def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, chat_model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = chat_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, chat_model)
    res = getResponse(ints, intents)
    return res


app = Flask(__name__)
app.static_folder = 'static'

app.secret_key = "your_secret_key"

model = joblib.load("train_model.pkl")

scaler = StandardScaler()

firebase_config = {
    # 'apiKey': "...",
    'apiKey': "AIzaSyBNHqLuaeZ_k8RWaG4jSh3QwWgZrbnLdX4",
    'authDomain': "groovy-scarab-381816.firebaseapp.com",
    'projectId': "groovy-scarab-381816",
    'storageBucket': "groovy-scarab-381816.appspot.com",
    'messagingSenderId': "1092611679235",
    'appId': "1:1092611679235:web:5368cb18b8d232a1721a36",
    'measurementId': "G-Y09T2RDLED",
    'databaseURL': ''
}

firebase = pyrebase.initialize_app(firebase_config)

# Initialize Firebase Authentication
auth = firebase.auth()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("unauth"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = user
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.create_user_with_email_and_password(email, password)
            session["user"] = user
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("register.html", error="Registration failed.")
    return render_template("register.html")

@app.route("/unauth")
def unauth():
    return "UnAuthorized 404 :x"


@app.route('/test')
@login_required
def test():
    return render_template('test.html')


@app.route('/submit', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        gender = request.form['gender']
        if (gender == "Female"):
            gender_no = 1
        else:
            gender_no = 2
        age = request.form['age']
        openness = request.form['openness']
        neuroticism = request.form['neuroticism']
        conscientiousness = request.form['conscientiousness']
        agreeableness = request.form['agreeableness']
        extraversion = request.form['extraversion']
        result = np.array([gender_no, age, openness, neuroticism,
                          conscientiousness, agreeableness, extraversion], ndmin=2)
        final = scaler.fit_transform(result)
        personality = str(model.predict(final)[0])
        return render_template("submit.html", answer=personality)

####################################################################


def remove_noise(text):
    tokens = word_tokenize(text)
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        token = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', token)
        token = lemmatizer.lemmatize(token.lower())
        if len(token) > 1 and token not in stopwords.words('english'):
            clean_tokens.append(token)
    return clean_tokens


# Load the pre-trained models and vectorizers
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
cluster_centers = pickle.load(open("cluster_centers.pkl", "rb"))

# Load the dataset
df = pd.read_csv("job_skills.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")


@app.route("/job_recommend")
@login_required
def job_recommend():
    return render_template("job_recom.html")


@app.route("/get_recommendations", methods=["POST"])
@login_required
def get_recommendations():
    position = request.form["position"]

    # Preprocess the user-provided position
    cleaned_position = " ".join(remove_noise(position))

    # Vectorize the user position using the loaded tfidf_vectorizer
    vectorized_position = tfidf_vectorizer.transform([cleaned_position])

    # Calculate similarity between user position and available positions
    similarities = vectorized_position.dot(cluster_centers.T)

    # Get the indices of top recommendations based on similarity
    top_indices = similarities.argsort(axis=1)[:, ::-1][:, :3]

    # Retrieve the actual position titles for the recommendations
    recommendations = []
    for indices in top_indices:
        recs = []
        for idx in indices:
            recs.append(df.iloc[idx]["Title"])
        recommendations.append(recs)

    return render_template("recommendations.html", recommendations=recommendations)


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


@app.route("/job_recommendation")
def job_recommendation():
    return render_template("")


if __name__ == "__main__":
    app.run(debug=True)

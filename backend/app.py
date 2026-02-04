import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

model = joblib.load("../model/model.pkl")
tfidf = joblib.load("../model/tfidf.pkl")

stop_words = set(stopwords.words("english"))

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     words = text.split()
#     words = [word for word in words if word not in stop_words]
#     return " ".join(words)
#     # return " ".join([w for w in text.split() if w not in stop_words])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


#if it give a string means clean_text is okh
msg = "Hey are we meeting today"
print(clean_text(msg))


@app.route("/predict", methods=["POST"])
def predict():
    msg = request.json["message"]
    clean_msg = clean_text(msg)
    vec = tfidf.transform([clean_msg])
    vec = tfidf.transform([clean_text(msg)])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = round(max(prob) * 100, 2)
    # return jsonify({"result": "Spam" if pred else "Not Spam"})
    return jsonify({
    "result": "Spam" if pred == "spam" else "Not Spam",
    "confidence": round(confidence * 100, 2)
})
    # return jsonify({"result": "Spam" if pred == "spam" else "Not Spam"})


print(tfidf.get_feature_names_out()[:20])


if __name__ == "__main__":
    app.run(debug=True)

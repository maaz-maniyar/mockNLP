import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


with open("data/intents.json") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

pickle.dump((model, vectorizer), open("model/intent_model.pkl", "wb"))
print("Model trained and saved to model/intent_model.pkl")
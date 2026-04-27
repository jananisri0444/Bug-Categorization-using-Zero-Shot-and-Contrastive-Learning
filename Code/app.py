from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from model import SiameseNetwork

app = Flask(__name__)

# Load data and models
df = pd.read_csv("labels.csv")  # Must have "Issue Category" column
df["Label"] = df["Issue Category"].astype("category").cat.codes
label_to_category = dict(enumerate(df["Issue Category"].astype("category").cat.categories))

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

siamese_model = SiameseNetwork()
siamese_model.load_state_dict(torch.load("siamese_model2.pth", map_location=torch.device("cpu")))
siamese_model.eval()

mlp_model = joblib.load("mlp_model2.pkl")
scaler = joblib.load("scaler.pkl")  # This must be saved during training

@app.route('/')
def index():
    return render_template("UI_HTML.html")

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    description = data.get("description", "")

    # Step 1: SBERT Embedding
    emb = sbert_model.encode([description])
    emb = torch.tensor(emb, dtype=torch.float32)

    # Step 2: Siamese Encoding
    with torch.no_grad():
        emb = siamese_model(emb).numpy()

    # Step 3: Scaling
    emb = scaler.transform(emb)

    # Step 4: Prediction
    pred = mlp_model.predict(emb)[0]
    category = label_to_category[pred]

    return jsonify({"category": category})

if __name__ == '__main__':
    app.run(debug=True)

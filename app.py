from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

# Load dataset
DATASET_PATH = "studios.csv"

def load_dataset():
    return pd.read_csv(DATASET_PATH)

def save_dataset(df):
    df.to_csv(DATASET_PATH, index=False)

# Thompson Sampling
def thompson_sampling_recommendations(df, n=5):
    recommendations = []
    for _, row in df.iterrows():
        # Generate a sample from Beta distribution
        score = np.random.beta(row['alpha'], row['beta'])
        recommendations.append((row['id'], score))
    # Sort by score and select top n
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
    return [rec[0] for rec in recommendations]

@app.route('/')
def home():
    # Load all studios
    df = load_dataset()

    # Generate recommendations
    recommended_ids = thompson_sampling_recommendations(df)
    recommendations = df[df['id'].isin(recommended_ids)].to_dict(orient='records')

    # Convert all studios to dictionary
    all_studios = df.to_dict(orient='records')

    return render_template('home.html', recommendations=recommendations, all_studios=all_studios)


@app.route('/click', methods=['POST'])
def click():
    # Handle user clicks
    studio_id = int(request.json.get('studio_id'))
    df = load_dataset()

    # Update alpha for clicked studio
    df.loc[df['id'] == studio_id, 'alpha'] += 1

    # Update beta for studios that were not clicked
    df.loc[df['id'] != studio_id, 'beta'] += 1

    # Save updated dataset
    save_dataset(df)
    return jsonify({'message': 'Feedback received'})

if __name__ == '__main__':
    app.run(debug=True)

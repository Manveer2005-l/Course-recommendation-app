import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Assume you have user_course_matrix, nmf_pred, nn_model loaded globally
import os, pickle

from tensorflow.keras.models import load_model

# Get absolute path of this folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load user_course_matrix
user_course_matrix_path = os.path.join(BASE_DIR, "user_course_matrix.pkl")
with open(user_course_matrix_path, "rb") as f:
    user_course_matrix = pickle.load(f)

weighted_features_path = os.path.join(BASE_DIR, "weighted_features.pkl")
with open(weighted_features_path, "rb") as f:
    weighted_features = pickle.load(f)

# Load user_course_matrix
df_path = os.path.join(BASE_DIR, "df.pkl")
with open(df_path, "rb") as f:
    df = pickle.load(f)


# Load NN model
nn_model_path = os.path.join(BASE_DIR, "nn_model.h5")
nn_model = load_model(nn_model_path, compile=False)

# Load NMF predictions
nmf_pred_path = os.path.join(BASE_DIR, "nmf_pred.pkl")
with open(nmf_pred_path, "rb") as f:
    nmf_pred = pickle.load(f)

def knn_predict_fn(user_label, top_n=10, return_full=False):
    """Return KNN predictions for a given user."""
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(user_course_matrix)

    user_idx = list(user_course_matrix.index).index(user_label)
    user_vector = user_course_matrix.iloc[user_idx].values.reshape(1, -1)

    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=6)  
    neighbor_pos = indices.flatten()[1:]

    neighbor_ratings = user_course_matrix.iloc[neighbor_pos]
    preds = neighbor_ratings.mean(axis=0).values  

    rated_mask = user_course_matrix.iloc[user_idx].values > 0
    preds[rated_mask] = -np.inf

    if return_full:
        return preds

    top_items = preds.argsort()[-top_n:][::-1]
    return user_course_matrix.columns[top_items], preds[top_items]


def nmf_pred_fn(user_label, top_n=10, return_full=False):
    """Return NMF predictions for a given user."""
    preds = nmf_pred.loc[user_label].values

    user_idx = list(user_course_matrix.index).index(user_label)
    rated_mask = user_course_matrix.iloc[user_idx].values > 0
    preds[rated_mask] = -np.inf

    if return_full:
        return preds

    top_items = preds.argsort()[-top_n:][::-1]
    return user_course_matrix.columns[top_items], preds[top_items]


def nn_predict_fn(user_label, top_n=10, return_full=False):
    """Return Neural Network predictions for a given user."""
    user_idx = list(user_course_matrix.index).index(user_label)
    n_items = user_course_matrix.shape[1]

    user_array = np.full(n_items, user_idx)
    item_array = np.arange(n_items)

    preds = nn_model.predict([user_array, item_array], verbose=0).flatten()
    preds = (preds * 2.0) + 3.0  

    rated_mask = user_course_matrix.iloc[user_idx].values > 0
    preds[rated_mask] = -np.inf

    if return_full:
        return preds

    top_items = preds.argsort()[-top_n:][::-1]
    return user_course_matrix.columns[top_items], preds[top_items]


def hybrid_recommend(user_label, k=10, weights=(0.25, 0.25, 0.25)):
    """Hybrid recommender combining KNN, NMF, and NN predictions."""
    w_knn, w_nmf, w_nn  = weights

    # Get full vectors from each model
    preds_knn = knn_predict_fn(user_label, return_full=True)
    preds_nmf = nmf_pred_fn(user_label, return_full=True)
    preds_nn  = nn_predict_fn(user_label, return_full=True)
    

    # Weighted average
    hybrid_preds = (w_knn * preds_knn) + (w_nmf * preds_nmf) + (w_nn * preds_nn) 

    user_idx = list(user_course_matrix.index).index(user_label)
    rated_mask = user_course_matrix.iloc[user_idx].values > 0
    hybrid_preds[rated_mask] = -np.inf

    top_items = np.argsort(hybrid_preds)[-k:][::-1]
    top_scores = hybrid_preds[top_items]

    return pd.Series(top_scores, index=user_course_matrix.columns[top_items])

from sklearn.metrics.pairwise import cosine_similarity

def course_similarity_fn(course_title, top_n=10, threshold=0.5):
    """
    Recommend courses based on course–course similarity.
    
    Args:
        user_label (str): The user identifier (e.g., "User_5").
        top_n (int): Number of recommendations.
        threshold (float): Minimum similarity between courses (0–1).
    
    Returns:
        pd.Series: Top-N recommended courses with similarity scores.
    """
    # --- Step 1: build similarity matrix (items × items) ---
    sim_matrix = cosine_similarity(weighted_features.astype(np.float32))  
    sim_df = pd.DataFrame(sim_matrix,
                          index=df['course_title'],
                          columns=df['course_title'])

    # --- Step 2: find courses this user already rated ---
    if course_title not in sim_df.index:
        raise ValueError(f"{course_title} not found in course list!")
    sim_scores = sim_df[course_title].drop(course_title)  # drop itself
    
    # --- Return top-N ---
    top_items = sim_scores.sort_values(ascending=False).head(top_n)
    return top_items 
    



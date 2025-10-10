import streamlit as st
from backend import knn_predict_fn, nmf_pred_fn, nn_predict_fn, hybrid_recommend , course_similarity_fn , user_course_matrix , weighted_features
import pandas as pd

st.set_page_config(page_title='Course Recommender',layout='centered')

st.title('Personalized course recommendation system')

st.sidebar.header('SETTINGS')

model_selection=st.sidebar.selectbox(
    'Select a Model',
    ['KNN','NMF','Neural Network','Course Similarity','Hybrid']
)

top_n=st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

user_id = st.text_input("Enter User ID (e.g., User_5):")

# Weights for Hybrid
weights = None
if model_selection == "Hybrid":
    st.sidebar.subheader("Hybrid Weights")
    w_knn = st.sidebar.slider("KNN Weight", 0.0, 1.0, 0.6, 0.05)
    w_nmf = st.sidebar.slider("NMF Weight", 0.0, 1.0, 0.35, 0.05)
    w_nn  = st.sidebar.slider("NN Weight", 0.0, 1.0, 0.05, 0.05)
    
    total = w_knn + w_nmf + w_nn 
    if total != 1.0:
        st.sidebar.warning("⚠️ Weights should sum to 1. Adjust sliders.")
    weights = (w_knn, w_nmf, w_nn )

# Recommendation button
# Course Similarity special handling
if model_selection == "Course Similarity":
    course_title = st.selectbox("Select a course", user_course_matrix.columns)
    
    if st.button("🔮 Recommend Courses"):
        try:
            res_series = course_similarity_fn(course_title, top_n=top_n)
            # Convert Series → DataFrame for display
            res_df = pd.DataFrame({
                "Course": res_series.index,
                "Similarity Score": res_series.values
            })
            st.subheader("🎯 Top Similar Courses")
            st.table(res_df)
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # For all other models
    if st.button("🔮 Recommend Courses"):
        if not user_id:
            st.error("Please enter a valid user ID (e.g., User_5).")
        else:
            try:
                if model_selection == "KNN":
                    courses, scores = knn_predict_fn(user_id, top_n)
                elif model_selection == "NMF":
                    courses, scores = nmf_pred_fn(user_id, top_n)
                elif model_selection == "Neural Network":
                    courses, scores = nn_predict_fn(user_id, top_n)
                elif model_selection == "Hybrid":
                    results = hybrid_recommend(user_id, k=top_n, weights=weights)
                    courses, scores = results.index, results.values

                st.subheader("🎯 Top Recommendations")
                for course, score in zip(courses, scores):
                    st.write(f"- {course} → **{score:.2f}**")

            except Exception as e:
                st.error(f"Error: {str(e)}")
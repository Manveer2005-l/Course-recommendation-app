# 🎓 Course Recommendation System

A machine learning-based system that recommends personalized courses to users using multiple models: **KNN**, **NMF**, **Neural Network**, and a **Hybrid Recommender**.

## 🚀 Features
- Trained and evaluated models using Precision, Recall, RMSE, and MAE
- Calculated novelty metric — *Average New/Unseen Courses per User*
- Implemented clustering (KMeans) and PCA visualizations
- Integrated a Streamlit web app for interactive course recommendations

## 🧠 Tech Stack
- **Language:** Python 3.13  
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Streamlit  
- **IDE:** Visual Studio Code / Jupyter Notebook

## 📊 Key Results
| Model | Precision | Recall | RMSE | MAE | Avg New/Unseen Courses |
|--------|------------|--------|------|------|------------------------|
| KNN | 0.10 | 0.25 | 0.23 | 0.05 | 3.4 |
| NMF | 0.12 | 0.31 | 0.23 | 0.06 | 4.1 |
| Neural Network | 0.00 | 0.01 | 0.78 | 0.64 | 2.7 |
| Hybrid | 0.14 | 0.33 | 0.22 | 0.05 | 4.8 |

## 🧩 How to Run
```bash
streamlit run recommender_app.py

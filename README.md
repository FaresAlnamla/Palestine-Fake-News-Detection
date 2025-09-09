# 📰 Fake News Detection using NLP & Machine Learning

## 🧾 Abstract & Introduction
The increasing spread of misinformation in digital media presents a major challenge. 
This project aims to develop a machine learning model that classifies **Arabic news articles** as either **real** or **fake**, using natural language processing (NLP) techniques. 
Our goal is to automate and enhance media verification by leveraging text-based features.


## 🗂️ Dataset Overview
- **Rows**: 5,352 news articles  
- **Columns**:
  - 🆔 **Id**: Unique identifier  
  - 🗓️ **date**: Date of publication  
  - 🌐 **platform**: News source (e.g., *Aljazeera*)  
  - 📰 **title**: Article headline  
  - 📄 **News content**: Full news body (*Arabic*)  
  - 🧪 **Label**: Either *real* or *fake*  

### Label Distribution
- ✅ **Real**: 3,913 articles  
- ⚠️ **Fake**: 1,439 articles  

### Top 3 Platforms + Others
| 🌐 Platform   | 📈 Number of Articles |
|--------------|------------------------|
| 🟢 Aljazeera  | 3,422 |
| 🔵 Misbar     | 1,426 |
| 🟣 Tibyan     | 247 |
| ⚪ Other      | 257 |

> 💡 Note: "Other" includes several smaller news channels grouped together.

### Data Quality Checks
- ✅ Total duplicate rows: 0  
- ✅ No null values in any column  

### Title Length Stats
| Metric | Value |
|--------|-------|
| 🔢 Mean | 55.76 |
| 🔽 Min  | 7     |
| 🔼 Max  | 379   |

### News Content Length Stats
| Metric | Value    |
|--------|----------|
| 🔢 Mean | 1,363.51 |
| 🔽 Min  | 7        |
| 🔼 Max  | 64,878   |


## 📊 Dataset Analysis

### Structure
- Preprocessed dataset contains 2 columns: `processed_text` (tokenized text) and `Label`  
- All rows contain tokenized Arabic text stored as stringified lists  

### Cleaning Steps
1. Converted token lists to plain text  
2. Removed Arabic stopwords  
3. Normalized words (removed punctuation & diacritics)  
4. Applied **TF-IDF vectorization** with up to 5,000 features using unigram + bigram  

### Distribution
- **Real**: 3,913 articles (~73%)  
- **Fake**: 1,439 articles (~27%)  

> ⚠️ Dataset is imbalanced — may affect recall for fake news.



## 💭 Opinion on Dataset Quality
The dataset is fairly clean and well-labeled.  
Tokenization and stopword removal improved preprocessing.  
The imbalance remains a challenge but can be addressed in future work with SMOTE or class weighting.



## 🔁 Methodology Flow

We trained multiple ML models:
- **Naive Bayes** → Lightweight, efficient with word counts  
- **Logistic Regression** → Strong linear baseline  
- **Linear SVM** → Performs well with sparse TF-IDF features  
- **Random Forest** → Captures non-linear patterns & avoids overfitting  



## 📈 Evaluation Results

| Model               | Accuracy | F1 (Fake) | F1 (Real) |
|---------------------|----------|-----------|-----------|
| Naive Bayes         | 87.4%    | 75.4%     | 91.5%     |
| Logistic Regression | 87.6%    | 74.9%     | 91.7%     |
| Linear SVM          | 87.9%    | 77.1%     | 91.7%     |
| **Random Forest**   | **88.8%**| **79.4%** | **92.3%** |

### ✅ Best Model
- **Random Forest Classifier** (`n_estimators=100, random_state=42`)  
- Achieved highest overall accuracy (88.8%)  
- Balanced performance across fake & real labels  



## 🗣️ Discussion
- Ensemble methods (Random Forest) outperformed linear models.  
- TF-IDF bigram features improved sensitivity to context.  
- Imbalance impacted minority class (fake) detection.  



## ✅ Conclusion & Future Work
We successfully built a machine learning pipeline that classifies Arabic news with ~89% accuracy.

**Future Improvements:**
- Apply resampling techniques (SMOTE, class-weighted models)  
- Experiment with **deep learning** (e.g., BERT for Arabic)  
- Deploy as a **real-time fake news detection app**  



## ⚙️ Installation & Usage

### Clone Repository
```bash
git clone https://github.com/FaresAlnamla/Palestine-Fake-News-Detection.git
cd Palestine-Fake-News-Detection
```


### Run Jupyter Notebook
```bash
jupyter notebook
```

Open `Fake News Detection Model.ipynb` and execute cells step by step.



## 📦 Dependencies
- Python 3.11.9 
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn, plotly  
- Jupyter Notebook  



## 📎 Appendices
- 📁 Dataset: `cleaned_news_dataset.csv`  
- 📄 Notebook: `Fake News Detection Model.ipynb`  
- 📊 Vectorizer: `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`  
- 📤 Best Model: `RandomForestClassifier(n_estimators=100, random_state=42)`  


✨ Built with data & code ❤️  

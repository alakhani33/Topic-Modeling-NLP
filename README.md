
# Topic Modeling of Customer Feedback

This project applies **topic modeling techniques** to analyze customer feedback data using both **Latent Dirichlet Allocation (LDA)** and **Non-Negative Matrix Factorization (NMF)**. The goal is to uncover latent topics from unstructured textual customer comments, providing actionable insights into customer sentiments, issues, and experiences.

## 📂 Project Overview

1. **Data Loading & Preprocessing**
   - A CSV file containing customer feedback was loaded.
   - Text cleaning steps included:
     - Removal of digits and non-alphabetic characters.
     - Lowercasing and whitespace normalization.
     - Stopwords removal using Scikit-learn’s built-in English stopword list.

2. **Feature Extraction**
   - Used `CountVectorizer` with:
     - `max_df=0.95` to exclude very frequent words
     - `min_df=10` to exclude very rare words
     - `stop_words='english'` for standard stopword filtering
   - Transformed customer comments into a document-term matrix.

3. **Topic Modeling Techniques**
   - Applied **Latent Dirichlet Allocation (LDA)** to the matrix.
   - Applied **Non-Negative Matrix Factorization (NMF)** as an alternative technique.
   - For both models:
     - Explored different numbers of topics (default set to 10).
     - Displayed top keywords for each discovered topic.

4. **Evaluation & Comparison**
   - Compared interpretability of LDA vs. NMF topics.
   - Highlighted differences in topic coherence and keyword groupings.

---

## 🧰 Dependencies

Make sure you have the following Python packages installed:

```bash
numpy
pandas
scikit-learn
```

You can install them using:

```bash
pip install numpy pandas scikit-learn
```

---

## 📊 Results Interpretation

Each model (LDA and NMF) outputs:
- A set of topics
- Top 10 keywords per topic

Use these keywords to infer:
- Major themes in customer feedback
- Common complaints or praises
- Areas needing improvement

---

## 📌 Example Output

Sample topic from LDA:

```
Topic 1: ['customer', 'service', 'call', 'issue', 'time', 'help', 'phone', 'problem', 'support', 'agent']
```

Sample topic from NMF:

```
Topic 3: ['billing', 'charge', 'account', 'payment', 'refund', 'amount', 'credit', 'fee', 'balance', 'statement']
```

---

## 📝 Notes

- This project uses **count-based vectorization**; you may explore **TF-IDF** for more nuanced weighting.
- Both **LDA** and **NMF** are unsupervised; manual interpretation of keywords is required to label topics meaningfully.

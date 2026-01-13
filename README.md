# Steam Reviews Analysis Pipeline with MLflow

Complete pipeline for sentiment classification, product clustering, and review summarization with MLflow tracking.

NOTE: This repository is currently a Work in Progress (WIP) / experimental / proof-of-concept. Core features are implemented, but several cases and auxiliary modules are not functional.

## 📋 Prerequisites

- Python 3.8+
- SQLite database with labeled reviews
- MLflow server with authentication enabled
- GPU recommended (but not required)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.template` to `.env` and update with your credentials:

```bash
cp .env.template .env
nano .env
```

Update these values:
- `MLFLOW_TRACKING_URI`: Your MLflow server URL
- `MLFLOW_TRACKING_USERNAME`: Your MLflow username
- `MLFLOW_TRACKING_PASSWORD`: Your MLflow password
- `DB_PATH`: Path to your SQLite database
- `TABLE_NAME`: Name of your reviews table

### 3. Prepare Your Database

Your SQLite database should have a table with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `review_text` | TEXT | The review content |
| `sentiment_label` | INTEGER/TEXT | 0/1 or 'negative'/'positive' |
| `product_category` | TEXT | Product category |
| `product_name` | TEXT | Product name |
| `rating` | REAL | Product rating (optional) |

**Example database structure:**

```sql
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    review_text TEXT NOT NULL,
    sentiment_label INTEGER,  -- 0 = negative, 1 = positive
    product_category TEXT,
    product_name TEXT,
    rating REAL
);
```

### 4. Run the Pipeline

```bash
python complete_mlflow_sentiment_pipeline.py
```

## 📊 What the Pipeline Does

### 1. Sentiment Classification
- Fine-tunes a transformer model (DistilBERT by default)
- Classifies reviews as positive or negative
- Logs metrics: accuracy, precision, recall, F1-score
- Generates confusion matrix
- Saves model to MLflow

### 2. Product Category Clustering
- Groups products into 4-6 meta-categories
- Uses TF-IDF + K-Means clustering
- Analyzes and names clusters
- Visualizes distribution
- Saves clustering model to MLflow

### 3. Review Summarization
- Generates article-style summaries for each category
- Identifies top 3 products per category
- Extracts main complaints
- Highlights worst product to avoid
- Uses BART or T5 for text generation

## 🎯 Expected Output

### MLflow UI
Navigate to your MLflow server to see:
- **Experiments**: All runs organized by task
- **Metrics**: Accuracy, precision, recall, F1-scores
- **Artifacts**: 
  - Confusion matrices
  - Cluster visualizations
  - Product summaries (Markdown files)
- **Models**: Trained sentiment and clustering models

### Console Output
```
============================================================
STEAM REVIEWS ANALYSIS PIPELINE
============================================================

[1/4] Loading data...
Loaded 10000 reviews

[2/4] Running sentiment classification...
Training distilbert-base-uncased on cuda...
Overall Accuracy: 94.50%

[3/4] Running category clustering...
Clustering products into 5 meta-categories...
Cluster 0 (2341 products): E-Readers, Tablets

[4/4] Running review summarization...
Generating summary for Category_0...

============================================================
PIPELINE COMPLETED SUCCESSFULLY!
============================================================
```

## 📈 Model Performance

Expected performance (depends on your dataset):

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| DistilBERT | 90-95% | ~30 min |
| BERT | 91-96% | ~60 min |
| RoBERTa | 92-97% | ~90 min |

*Times based on 10K reviews with GPU*

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in the code:
```python
batch_size=8  # instead of 16
```

### MLflow Authentication Failed
Check your credentials in `.env`:
```bash
curl -u admin:password http://databricks.domain.com/health
```

### Database Not Found
Verify the path:
```bash
ls -la reviews.db
```

### Module Not Found
Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

## 📂 Project Structure

```
.
├── complete_mlflow_sentiment_pipeline.py  # Main pipeline
├── requirements.txt                       # Dependencies
├── .env.template                          # Environment template
├── .env                                   # Your credentials (gitignored)
├── reviews.db                             # Your SQLite database
├── results/                               # Training outputs
├── logs/                                  # Training logs
├── confusion_matrix.png                   # Generated plots
├── cluster_distribution.png
└── summary_Category_*.md                  # Generated summaries
```

## 🔐 Security Notes

1. **Never commit `.env` file** - Add to `.gitignore`:
   ```
   .env
   *.db
   results/
   logs/
   ```

2. **Use strong passwords** in your `.env` file

3. **Restrict MLflow access** to authorized users only

## 📚 Model Options

### Sentiment Classification Models

| Model | Pros | Cons |
|-------|------|------|
| `distilbert-base-uncased` | Fast, lightweight | Slightly lower accuracy |
| `bert-base-uncased` | Good balance | Slower than DistilBERT |
| `roberta-base` | High accuracy | Slower, more memory |
| `cardiffnlp/twitter-roberta-base-sentiment` | Great for short text | Specialized |

### Summarization Models

| Model | Pros | Cons |
|-------|------|------|
| `facebook/bart-large-cnn` | Excellent summaries | Large model |
| `google/flan-t5-base` | Versatile | May need more tuning |
| `google/flan-t5-large` | Best quality | Very slow |

## 🤝 Contributing

Feel free to modify the pipeline for your specific needs!

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review MLflow logs
3. Verify database schema matches expectations

## 📄 License

This template is provided as-is for educational and commercial use.
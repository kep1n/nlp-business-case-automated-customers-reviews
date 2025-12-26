"""
Complete MLflow Pipeline for Amazon Reviews Analysis
- Sentiment Classification (Transformer-based)
- Product Category Clustering
- Review Summarization with Generative AI
"""
import logclass
import os
import sqlite3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.transformers
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, pipeline, Trainer, TrainingArguments
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# MLflow Configuration
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', 'admin')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', 'password')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Set MLflow experiment
mlflow.set_experiment("amazon-reviews-analysis")


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data_from_sqlite(db_path, table_name="reviews"):
    """Load labeled reviews from SQLite database"""
    print("Loading data from SQLite...")
    conn = sqlite3.connect(db_path)
    
    # Adjust column names according to your database schema
    query = f"""
    SELECT 
        review_text,
        sentiment_label,
        product_category,
        product_name,
        rating
    FROM {table_name}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSentiment distribution:\n{df['sentiment_label'].value_counts()}")
    print(f"\nCategory distribution:\n{df['product_category'].value_counts()}")
    
    return df


# ============================================================================
# 2. SENTIMENT CLASSIFICATION WITH TRANSFORMERS
# ============================================================================

class SentimentClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        """
        Initialize sentiment classifier
        model_name options:
        - distilbert-base-uncased (lightweight, fast)
        - bert-base-uncased (strong general purpose)
        - roberta-base (robust to nuances)
        - cardiffnlp/twitter-roberta-base-sentiment (for short texts)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def prepare_data(self, texts, labels):
        """Prepare data for training"""
        dataset = Dataset.from_dict({
            'text': texts,
            'label': labels
        })
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize texts"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    def train(self, train_texts, train_labels, val_texts, val_labels, 
              epochs=3, batch_size=16, learning_rate=2e-5):
        """Fine-tune transformer model"""
        
        print(f"\nTraining {self.model_name} on {self.device}...")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        ).to(self.device)
        
        # Prepare datasets
        train_dataset = self.prepare_data(train_texts, train_labels)
        val_dataset = self.prepare_data(val_texts, val_labels)
        
        # Tokenize
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        trainer.train()
        
        return trainer
    
    def predict(self, texts):
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
                predictions.append(pred)
        
        return np.array(predictions)


def evaluate_sentiment_model(y_true, y_pred, class_names=['Negative', 'Positive']):
    """Comprehensive model evaluation"""
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")
    
    print("Per-Class Metrics:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision[i]*100:.2f}%")
        print(f"  Recall:    {recall[i]*100:.2f}%")
        print(f"  F1-Score:  {f1[i]*100:.2f}%")
        print()
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'cm_plot_path': cm_path
    }


def run_sentiment_classification(df, model_name="distilbert-base-uncased"):
    """Complete sentiment classification pipeline with MLflow tracking"""
    
    with mlflow.start_run(run_name=f"sentiment_{model_name}"):
        
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "sentiment_classification")
        mlflow.log_param("total_samples", len(df))
        
        # Prepare data
        X = df['review_text'].tolist()
        y = df['sentiment_label'].values
        
        # Convert labels to binary if needed (0=negative, 1=positive)
        if y.dtype == object:
            y = (y == 'positive').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Initialize and train model
        classifier = SentimentClassifier(model_name=model_name, num_labels=2)
        
        # Train
        trainer = classifier.train(
            X_train, y_train.tolist(),
            X_test[:100], y_test[:100].tolist(),  # Use subset for validation
            epochs=3,
            batch_size=16
        )
        
        # Predict on test set
        print("\nEvaluating on test set...")
        y_pred = classifier.predict(X_test)
        
        # Evaluate
        metrics = evaluate_sentiment_model(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision_negative", metrics['precision'][0])
        mlflow.log_metric("precision_positive", metrics['precision'][1])
        mlflow.log_metric("recall_negative", metrics['recall'][0])
        mlflow.log_metric("recall_positive", metrics['recall'][1])
        mlflow.log_metric("f1_negative", metrics['f1'][0])
        mlflow.log_metric("f1_positive", metrics['f1'][1])
        
        # Log artifacts
        mlflow.log_artifact(metrics['cm_plot_path'])
        
        # Log model
        mlflow.transformers.log_model(
            transformers_model={
                "model": classifier.model,
                "tokenizer": classifier.tokenizer
            },
            artifact_path="sentiment_model"
        )
        
        print(f"\n✓ Sentiment classification completed!")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
        
        return classifier, metrics


# ============================================================================
# 3. PRODUCT CATEGORY CLUSTERING
# ============================================================================

class CategoryClusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = None
        
    def fit(self, product_descriptions):
        """Fit clustering model"""
        print(f"\nClustering products into {self.n_clusters} meta-categories...")
        
        # Vectorize product descriptions
        X = self.vectorizer.fit_transform(product_descriptions)
        
        # Cluster
        self.cluster_labels = self.kmeans.fit_predict(X)
        
        return self.cluster_labels
    
    def analyze_clusters(self, df, cluster_labels):
        """Analyze and name clusters"""
        df['meta_category'] = cluster_labels
        
        print("\nCluster Analysis:")
        print("="*60)
        
        cluster_info = {}
        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['meta_category'] == cluster_id]
            top_categories = cluster_df['product_category'].value_counts().head(5)
            top_products = cluster_df['product_name'].value_counts().head(3)
            
            print(f"\nCluster {cluster_id} ({len(cluster_df)} products):")
            print(f"  Top categories: {', '.join(top_categories.index.tolist())}")
            print(f"  Sample products: {', '.join(top_products.index.tolist()[:3])}")
            
            cluster_info[f"cluster_{cluster_id}"] = {
                'size': len(cluster_df),
                'top_categories': top_categories.to_dict(),
                'top_products': top_products.to_dict()
            }
        
        # Suggest meta-category names
        suggested_names = self.suggest_cluster_names(cluster_info)
        
        return cluster_info, suggested_names
    
    def suggest_cluster_names(self, cluster_info):
        """Suggest meaningful names for clusters"""
        # This is a placeholder - you should analyze and manually assign names
        # based on the products in each cluster
        suggestions = {}
        for cluster_id in range(self.n_clusters):
            suggestions[cluster_id] = f"Meta-Category-{cluster_id}"
        
        print("\n" + "="*60)
        print("SUGGESTED META-CATEGORY NAMES:")
        print("="*60)
        print("Review the cluster analysis above and assign meaningful names")
        print("Examples: 'E-Readers', 'Batteries & Power', 'Accessories',")
        print("         'Pet Products', 'Home Electronics', etc.")
        
        return suggestions


def run_category_clustering(df, n_clusters=5):
    """Complete clustering pipeline with MLflow tracking"""
    
    with mlflow.start_run(run_name=f"clustering_{n_clusters}_categories"):
        
        # Log parameters
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("task", "category_clustering")
        
        # Create product descriptions for clustering
        df['product_desc'] = df['product_category'] + ' ' + df['product_name']
        
        # Fit clustering
        clusterer = CategoryClusterer(n_clusters=n_clusters)
        cluster_labels = clusterer.fit(df['product_desc'].tolist())
        
        # Analyze clusters
        cluster_info, suggested_names = clusterer.analyze_clusters(df, cluster_labels)
        
        # Log cluster sizes
        for cluster_id in range(n_clusters):
            mlflow.log_metric(f"cluster_{cluster_id}_size", 
                            cluster_info[f"cluster_{cluster_id}"]['size'])
        
        # Visualize clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_sizes = [cluster_info[f"cluster_{i}"]['size'] 
                        for i in range(n_clusters)]
        ax.bar(range(n_clusters), cluster_sizes)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Products')
        ax.set_title('Product Distribution Across Meta-Categories')
        plt.tight_layout()
        
        cluster_plot = 'cluster_distribution.png'
        plt.savefig(cluster_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(cluster_plot)
        
        # Log model
        mlflow.sklearn.log_model(clusterer.kmeans, "clustering_model")
        
        print(f"\n✓ Category clustering completed!")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
        
        return clusterer, df


# ============================================================================
# 4. REVIEW SUMMARIZATION WITH GENERATIVE AI
# ============================================================================

class ReviewSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize summarization model
        model_name options:
        - facebook/bart-large-cnn (excellent for summarization)
        - google/flan-t5-base (versatile, can follow instructions)
        - google/flan-t5-large (better quality, slower)
        """
        self.model_name = model_name
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def prepare_category_reviews(self, df, cluster_id):
        """Prepare reviews for a specific category"""
        cluster_df = df[df['meta_category'] == cluster_id]
        
        # Get top 3 products by number of reviews
        top_products = cluster_df['product_name'].value_counts().head(3).index.tolist()
        
        # Get reviews for each product
        product_reviews = {}
        product_complaints = {}
        
        for product in top_products:
            product_df = cluster_df[cluster_df['product_name'] == product]
            
            # Positive reviews
            positive = product_df[product_df['sentiment_label'] == 1]['review_text'].tolist()
            
            # Negative reviews (complaints)
            negative = product_df[product_df['sentiment_label'] == 0]['review_text'].tolist()
            
            product_reviews[product] = {
                'positive': positive[:10],  # Top 10
                'negative': negative[:10],
                'avg_rating': product_df['rating'].mean() if 'rating' in product_df.columns else None
            }
        
        # Find worst product
        product_ratings = cluster_df.groupby('product_name')['rating'].agg(['mean', 'count'])
        product_ratings = product_ratings[product_ratings['count'] >= 5]  # Min 5 reviews
        worst_product = product_ratings['mean'].idxmin()
        
        return {
            'top_products': top_products,
            'product_reviews': product_reviews,
            'worst_product': worst_product,
            'cluster_size': len(cluster_df)
        }
    
    def generate_summary(self, category_data, category_name):
        """Generate article summary for a category"""
        
        print(f"\nGenerating summary for {category_name}...")
        
        # Prepare content for summarization
        content_parts = []
        
        # Top products section
        content_parts.append(f"Top Products in {category_name}:\n")
        for i, product in enumerate(category_data['top_products'], 1):
            reviews = category_data['product_reviews'][product]
            avg_rating = reviews['avg_rating']
            
            # Summarize positive aspects
            if reviews['positive']:
                positive_text = ' '.join(reviews['positive'][:5])
                content_parts.append(f"{i}. {product} (Rating: {avg_rating:.1f}/5)")
                content_parts.append(f"Positive aspects: {positive_text[:500]}")
            
            # Summarize complaints
            if reviews['negative']:
                negative_text = ' '.join(reviews['negative'][:3])
                content_parts.append(f"Common complaints: {negative_text[:300]}")
        
        # Worst product section
        content_parts.append(f"\nProduct to Avoid: {category_data['worst_product']}")
        
        full_content = '\n'.join(content_parts)
        
        # Generate summary with the model
        try:
            summary = self.summarizer(
                full_content,
                max_length=500,
                min_length=200,
                do_sample=False
            )
            
            generated_text = summary[0]['summary_text']
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            generated_text = "Summary generation failed. Using template."
            generated_text = self._generate_template_summary(category_data, category_name)
        
        return generated_text
    
    def _generate_template_summary(self, category_data, category_name):
        """Fallback template-based summary"""
        
        summary = f"# {category_name} Product Recommendations\n\n"
        summary += "## Top 3 Products\n\n"
        
        for i, product in enumerate(category_data['top_products'], 1):
            reviews = category_data['product_reviews'][product]
            avg_rating = reviews['avg_rating'] or 0
            
            summary += f"### {i}. {product}\n"
            summary += f"**Rating:** {avg_rating:.1f}/5\n\n"
            
            if reviews['positive']:
                summary += "**Highlights:**\n"
                summary += f"- {reviews['positive'][0][:150]}...\n\n"
            
            if reviews['negative']:
                summary += "**Common Complaints:**\n"
                summary += f"- {reviews['negative'][0][:150]}...\n\n"
        
        summary += f"## Product to Avoid\n\n"
        summary += f"**{category_data['worst_product']}** - Based on customer reviews, "
        summary += "this product has received consistently low ratings.\n"
        
        return summary


def run_review_summarization(df, n_clusters=5):
    """Complete summarization pipeline with MLflow tracking"""
    
    with mlflow.start_run(run_name="review_summarization"):
        
        mlflow.log_param("task", "review_summarization")
        mlflow.log_param("n_clusters", n_clusters)
        
        # Initialize summarizer
        summarizer = ReviewSummarizer()
        
        # Generate summaries for each meta-category
        summaries = {}
        
        for cluster_id in range(n_clusters):
            category_name = f"Category_{cluster_id}"
            
            # Prepare data
            category_data = summarizer.prepare_category_reviews(df, cluster_id)
            
            # Generate summary
            summary = summarizer.generate_summary(category_data, category_name)
            
            summaries[category_name] = summary
            
            # Save summary to file
            summary_file = f'summary_{category_name}.md'
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            # Log to MLflow
            mlflow.log_artifact(summary_file)
            mlflow.log_text(summary, f"{category_name}_summary.txt")
            
            print(f"\n{category_name} Summary:")
            print("-" * 60)
            print(summary[:500] + "...\n")
        
        print(f"\n✓ Review summarization completed!")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")
        
        return summaries


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete analysis pipeline"""
    
    print("\n" + "="*60)
    print("AMAZON REVIEWS ANALYSIS PIPELINE")
    print("="*60)
    
    # Configuration
    DB_PATH = "reviews.db"  # Update with your database path
    TABLE_NAME = "reviews"  # Update with your table name
    N_CLUSTERS = 5  # Number of meta-categories
    
    # 1. Load Data
    print("\n[1/4] Loading data...")
    df = load_data_from_sqlite(DB_PATH, TABLE_NAME)
    
    # 2. Sentiment Classification
    print("\n[2/4] Running sentiment classification...")
    sentiment_model, sentiment_metrics = run_sentiment_classification(
        df,
        model_name="distilbert-base-uncased"  # Change model here if needed
    )
    
    # 3. Category Clustering
    print("\n[3/4] Running category clustering...")
    clusterer, df_clustered = run_category_clustering(df, n_clusters=N_CLUSTERS)
    
    # 4. Review Summarization
    print("\n[4/4] Running review summarization...")
    summaries = run_review_summarization(df_clustered, n_clusters=N_CLUSTERS)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nCheck MLflow UI at: {os.environ['MLFLOW_TRACKING_URI']}")
    print("All results, models, and artifacts have been logged.")
    

if __name__ == "__main__":
    main()
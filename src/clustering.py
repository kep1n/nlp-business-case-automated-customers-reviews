"""
Steam Reviews Clustering Pipeline

This module provides a complete pipeline for clustering Steam game reviews using
TF-IDF vectorization and K-Means clustering. It includes data loading, text preprocessing,
clustering, and visualization capabilities.
"""

import sqlite3
import re
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from mpl_toolkits.mplot3d import Axes3D
import logging
import logclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_LOGGER = 'ProjectLogger'


class DatabaseLoader:
    """
    Handles data loading operations from SQLite databases.

    This class provides methods to connect to SQLite databases and
    retrieve review data for further processing.
    """

    def __init__(self, db_path: str):
        """
        Initialize the DatabaseLoader with a database path.

        Args:
            db_path: Path to the SQLite database file

        Raises:
            FileNotFoundError: If the database file doesn't exist
        """
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            self.logger.error(f"Database file not found: {db_path}")
            raise FileNotFoundError

    def load_reviews(self, table_name: str, review_column: str = 'review') -> pd.DataFrame:
        """
        Load review data from the specified table.

        Args:
            table_name: Name of the table containing reviews
            review_column: Name of the column containing review text (default: 'review')

        Returns:
            DataFrame containing the review data

        Raises:
            sqlite3.Error: If there's an error executing the query
        """
        query = f"SELECT {review_column} FROM {table_name}"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                print(f"✓ Loaded {len(df)} reviews from {table_name}")
                return df
        except sqlite3.Error:
            self.logger.exception(f"Error loading database")
            raise


class TextPreprocessor:
    """
    Handles all text preprocessing operations for review data.

    This class provides methods for cleaning, normalizing, and transforming
    text data through various NLP techniques including stopword removal,
    stemming, and lemmatization.
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize the TextPreprocessor with NLP tools.

        Args:
            language: Language for stopwords (default: 'english')
        """
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Perform basic text cleaning operations.

        Operations performed:
        - Convert to lowercase
        - Remove numbers and punctuation
        - Remove single-character words
        - Normalize whitespace

        Args:
            text: Raw input text

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove numbers and punctuation
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove single-character words
        text = re.sub(r'\b\w{1}\b', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text: str) -> str:
        """
        Remove common stopwords from text using regex patterns.

        Args:
            text: Input text with stopwords

        Returns:
            Text with stopwords removed
        """
        # Create regex pattern with word boundaries for exact matches
        pattern = r'\b(' + '|'.join(self.stop_words) + r')\b'
        text = re.sub(pattern, '', text)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def stem_text(self, text: str) -> str:
        """
        Apply Porter stemming to reduce words to their root form.

        Args:
            text: Input text to stem

        Returns:
            Stemmed text
        """
        words = re.findall(r'\b\w+\b', text)
        return ' '.join([self.stemmer.stem(w) for w in words])

    def lemmatize_text(self, text: str) -> str:
        """
        Apply lemmatization to reduce words to their dictionary form.

        Args:
            text: Input text to lemmatize

        Returns:
            Lemmatized text
        """
        words = re.findall(r'\b\w+\b', text)
        return ' '.join([self.lemmatizer.lemmatize(w) for w in words])

    def preprocess(self, text: str, remove_stops: bool = True,
                   use_lemma: bool = True) -> str:
        """
        Execute complete preprocessing pipeline on input text.

        Pipeline steps:
        1. Basic cleaning (lowercase, remove punctuation/numbers)
        2. Stopword removal (optional)
        3. Stemming or lemmatization (optional)

        Args:
            text: Raw input text
            remove_stops: Whether to remove stopwords (default: True)
            use_lemma: Use lemmatization if True, stemming if False (default: True)

        Returns:
            Fully preprocessed text
        """
        # Basic cleaning
        self.logger.info('Performing basic cleaning...')
        text = self.clean_text(text)

        # Stopword removal
        self.logger.info('Performing stopwords cleaning...')
        if remove_stops:
            text = self.remove_stopwords(text)

        # Morphological normalization
        if use_lemma:
            text = self.lemmatize_text(text)
        else:
            text = self.stem_text(text)

        return text

    def preprocess_dataframe(self, df: pd.DataFrame,
                             text_column: str = 'review',
                             output_column: str = 'clean_review') -> pd.DataFrame:
        """
        Apply preprocessing to an entire DataFrame column.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text to preprocess
            output_column: Name of column to store preprocessed text

        Returns:
            DataFrame with added preprocessed text column
        """
        self.logger.info(f"Preprocessing {len(df)} reviews...")
        df[output_column] = df[text_column].apply(self.preprocess)
        self.logger.info(f"Preprocessing complete")
        return df


class ClusteringPipeline:
    """
    Manages the complete clustering pipeline including vectorization and clustering.

    This class handles TF-IDF vectorization of text data and K-Means clustering,
    providing methods to fit models and extract insights from clusters.
    """

    def __init__(self, n_clusters: int = 5, max_features: int = 1000,
                 ngram_range: Tuple[int, int] = (1, 3), random_state: int = 42):
        """
        Initialize the clustering pipeline with specified parameters.

        Args:
            n_clusters: Number of clusters for K-Means (default: 5)
            max_features: Maximum number of features for TF-IDF (default: 1000)
            ngram_range: Range of n-grams to extract (default: (1, 3))
            random_state: Random seed for reproducibility (default: 42)
        """
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range
        )

        # Initialize K-Means clustering
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        self.tfidf_matrix: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, texts: pd.Series) -> 'ClusteringPipeline':
        """
        Fit the clustering pipeline on input texts.

        Args:
            texts: Series of preprocessed text data

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Vectorizing text data...")
        # Transform text to TF-IDF vectors
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.logger.info(f"Created TF-IDF matrix: {self.tfidf_matrix.shape}")

        self.logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        # Fit K-Means clustering
        self.kmeans.fit(self.tfidf_matrix)
        self.logger.info(f"Clustering complete")

        return self

    def predict(self, df: pd.DataFrame, text_column: str = 'clean_review',
                cluster_column: str = 'cluster') -> pd.DataFrame:
        """
        Assign cluster labels to input data.

        Args:
            df: Input DataFrame
            text_column: Column containing preprocessed text
            cluster_column: Column name for cluster assignments

        Returns:
            DataFrame with added cluster labels
        """
        if self.tfidf_matrix is None:
            self.logger.error("Pipeline must be fitted before prediction")
            raise ValueError

        df[cluster_column] = self.kmeans.labels_
        self.logger.info(f"Assigned {self.n_clusters} cluster labels")

        return df

    def get_top_terms(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Extract top terms for each cluster based on centroid values.

        Args:
            n_terms: Number of top terms to extract per cluster (default: 10)

        Returns:
            Dictionary mapping cluster IDs to lists of top terms
        """
        if self.feature_names is None:
            self.logger.error("Pipeline must be fitted before extracting terms")
            raise ValueError

        # Sort centroids to find top terms
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]

        cluster_terms = {}
        for i in range(self.n_clusters):
            top_indices = order_centroids[i, :n_terms]
            cluster_terms[i] = [self.feature_names[ind] for ind in top_indices]

        return cluster_terms

    def print_cluster_analysis(self, n_terms: int = 10) -> None:
        """
        Print a formatted analysis of top terms per cluster.

        Args:
            n_terms: Number of top terms to display per cluster
        """
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS - Top Terms per Cluster")
        print("=" * 60)

        cluster_terms = self.get_top_terms(n_terms)

        for cluster_id, terms in cluster_terms.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  {', '.join(terms)}")

        print("\n" + "=" * 60)


class ClusterVisualizer:
    """
    Handles visualization of clustering results using dimensionality reduction.

    This class provides methods to create 2D and 3D visualizations of
    clustered data using PCA for dimensionality reduction.
    """

    def __init__(self, output_dir: str = '../results'):
        """
        Initialize the visualizer with output directory.

        Args:
            output_dir: Directory to save visualization outputs (default: '../results')
        """
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_3d(self, tfidf_matrix: np.ndarray, labels: np.ndarray,
                     cluster_centers: np.ndarray, filename: str = 'clustering_3d.png',
                     figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create a 3D visualization of clusters using PCA.

        Args:
            tfidf_matrix: TF-IDF matrix of documents
            labels: Cluster labels for each document
            cluster_centers: K-Means cluster centers
            filename: Output filename for the plot (default: 'clustering_3d.png')
            figsize: Figure size as (width, height) (default: (15, 10))
        """
        self.logger.info(f"Creating 3D visualization...")

        # Reduce dimensionality to 3D using PCA
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())
        reduced_centers = pca.transform(cluster_centers)

        # Explained variance
        explained_var = pca.explained_variance_ratio_
        self.logger.info(f"PCA explained variance: {explained_var.sum():.2%}")

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of data points
        scatter = ax.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            reduced_features[:, 2],
            c=labels,
            cmap='viridis',
            s=25,
            edgecolor='k',
            alpha=0.6,
            label='Reviews'
        )

        # Plot cluster centers
        ax.scatter(
            reduced_centers[:, 0],
            reduced_centers[:, 1],
            reduced_centers[:, 2],
            c='red',
            marker='X',
            s=200,
            edgecolor='black',
            linewidth=2,
            label='Centroids'
        )

        # Labels and formatting
        ax.set_title('Review Clusters (3D PCA Projection)', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)', fontsize=12)
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1%} var)', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Cluster ID', fontsize=12)

        ax.legend()

        # Save and display
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved 3D visualization to {output_path}")
        plt.show()
        plt.close()

    def visualize_2d(self, tfidf_matrix: np.ndarray, labels: np.ndarray,
                     filename: str = 'clustering_2d.png',
                     figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create a 2D visualization of clusters using PCA.

        Args:
            tfidf_matrix: TF-IDF matrix of documents
            labels: Cluster labels for each document
            filename: Output filename for the plot (default: 'clustering_2d.png')
            figsize: Figure size as (width, height) (default: (12, 8))
        """
        self.logger.info(f"Creating 2D visualization...")

        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(tfidf_matrix.toarray())

        # Create 2D plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=labels,
            cmap='viridis',
            s=30,
            edgecolor='k',
            alpha=0.6
        )

        plt.title('Review Clusters (2D PCA Projection)', fontsize=16, fontweight='bold')
        plt.xlabel('PCA Component 1', fontsize=12)
        plt.ylabel('PCA Component 2', fontsize=12)
        plt.colorbar(scatter, label='Cluster ID')

        # Save and display
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved 2D visualization to {output_path}")
        plt.show()
        plt.close()


class ReviewClusteringExperiment:
    """
    Main orchestrator class for the complete review clustering experiment.

    This class coordinates all components of the clustering pipeline and
    integrates with MLflow for experiment tracking.
    """

    def __init__(self, db_path: str, experiment_name: str = "steam-reviews-clustering"):
        """
        Initialize the experiment with database path and experiment name.

        Args:
            db_path: Path to SQLite database containing reviews
            experiment_name: Name for MLflow experiment (default: "steam-reviews-clustering")
        """
        self.logger = logging.getLogger(PROJECT_LOGGER)
        self.db_path = db_path
        self.experiment_name = experiment_name

        # Initialize components
        self.db_loader = DatabaseLoader(db_path)
        self.preprocessor = TextPreprocessor()
        self.pipeline: Optional[ClusteringPipeline] = None
        self.visualizer = ClusterVisualizer()

        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        mlflow.autolog()

    def run(self, table_name: str = 'reviews', n_clusters: int = 5,
            max_features: int = 1000) -> pd.DataFrame:
        """
        Execute the complete clustering experiment.

        Args:
            table_name: Name of the database table containing reviews
            n_clusters: Number of clusters to create (default: 5)
            max_features: Maximum features for TF-IDF (default: 1000)

        Returns:
            DataFrame with reviews and assigned cluster labels
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("table_name", table_name)

            print("\n" + "=" * 60)
            print("STEAM REVIEWS CLUSTERING EXPERIMENT")
            print("=" * 60 + "\n")

            # Step 1: Load data
            self.logger.info("Loading Data")
            df = self.db_loader.load_reviews(table_name)
            mlflow.log_metric("n_reviews", len(df))

            # Step 2: Preprocess text
            self.logger.info("Preprocessing Text")
            df = self.preprocessor.preprocess_dataframe(df)

            # Step 3: Clustering
            self.logger.info("Clustering...")
            self.pipeline = ClusteringPipeline(
                n_clusters=n_clusters,
                max_features=max_features
            )
            self.pipeline.fit(df['clean_review'])
            df = self.pipeline.predict(df)

            # Step 4: Analysis
            self.logger.info("Cluster Analysis")
            self.pipeline.print_cluster_analysis()

            # Step 5: Visualization
            self.logger.info("Visualization")
            self.visualizer.visualize_3d(
                self.pipeline.tfidf_matrix,
                df['cluster'].values,
                self.pipeline.kmeans.cluster_centers_
            )

            self.logger.info("EXPERIMENT COMPLETE")
            return df


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'project_log.log')
    logger = logclass.ProjectLogger(log_path=log_path).get_logger()
    """
    Main entry point for the clustering experiment.
    """
    # Configuration
    DB_PATH = r'E:\SynologyDrive\ironhack\week_14\day_2\nlp-business-case-automated-customers-reviews\data\raw\gamesDB.db'  # TODO: Add your database path here
    TABLE_NAME = 'reviews'
    N_CLUSTERS = 5
    MAX_FEATURES = 1000

    # Run experiment
    experiment = ReviewClusteringExperiment(DB_PATH)
    df_clustered = experiment.run(
        table_name=TABLE_NAME,
        n_clusters=N_CLUSTERS,
        max_features=MAX_FEATURES
    )

    # Optional: Save results
    output_path = Path('../results/clustered_reviews.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clustered.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()



# HDBSCAN + Embeddings (recommended)
# BERTopic (Embedding + HDBSCAN + c-TF-IDF)
# all-MiniLM-L6-v2
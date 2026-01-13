import torch
import sqlite3
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from typing import List, Dict, Union, Optional
from google.colab import userdata
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = userdata.get('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = userdata.get('username')
os.environ['MLFLOW_TRACKING_PASSWORD'] = userdata.get('password')

# Set MLflow experiment
mlflow.set_experiment("steam-reviews-clustering")
mlflow.autolog()

class SteamZeroShotClassifier:
    """
    A pipeline for classifying Steam reviews into semantic categories using Zero-Shot Classification.
    
    This approach allows us to categorize reviews without specific training data for these labels,
    leveraging a pre-trained NLI model.
    """
    
    def __init__(self, model_name: str = "valhalla/distilbart-mnli-12-6", device: int = -1, fp16: bool = True):
        """
        Initialize the Zero-Shot Classification pipeline. 
        
        Args:
            model_name: The Hugging Face model to use. Defaults to 'valhalla/distilbart-mnli-12-6' (faster).
            device: Device to run on. -1 for CPU, 0+ for GPU.
            fp16: Whether to use 16-bit precision (requires GPU). Defaults to True.
        """
        print(f"Loading Zero-Shot Classification pipeline with model: {model_name}...")
        
        # Check if CUDA is available and user didn't specify CPU explicitly
        if device == -1 and torch.cuda.is_available():
            device = 0
            print(f"CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU.")

        pipeline_kwargs = {}
        if device >= 0 and fp16:
            print("Enabling FP16 inference for speed.")
            pipeline_kwargs["torch_dtype"] = torch.float16

        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            **pipeline_kwargs
        )
        
        # Candidate labels derived from GEMINI.md for Horror Games
        self.candidate_labels = [
            "Atmosphere & Immersion",
            "Technical Performance", 
            "Gameplay Mechanics",
            "Stability & Bugs",
            "Story & Narrative", # Added as it's common in horror
            "Price & Value"      # Added as it's common in steam reviews
        ]
        print("Pipeline loaded successfully.")

    def classify_review(self, text: str, multi_label: bool = True) -> Dict:
        """
        Classify a single review text.
        
        Args:
            text: The review content.
            multi_label: Whether a review can belong to multiple categories. 
                         Steam reviews often mention multiple aspects (e.g. good graphics but buggy).
        
        Returns:
            Dictionary containing labels and scores.
        """
        try:
            if not text or not text.strip():
                return {}
            
            result = self.classifier(
                text, 
                self.candidate_labels, 
                multi_label=multi_label
            )
            return result
        except Exception as e:
            print(f"Error classifying review: {e}")
            return {}

    def classify_batch(self, texts: List[str], multi_label: bool = True) -> List[Dict]:
        """
        Classify a batch of reviews. 
        Handles empty strings to avoid pipeline errors.
        """
        # Filter valid texts and keep track of indices
        valid_inputs = []
        valid_indices = []
        
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_inputs.append(text)
                valid_indices.append(idx)
        
        if not valid_inputs:
            return [{} for _ in texts]

        # Run classification on valid inputs only
        results = self.classifier(valid_inputs, self.candidate_labels, multi_label=multi_label)
        
        # If pipeline returns a single dict (for 1 item), wrap it in a list
        if isinstance(results, dict):
            results = [results]

        # Reconstruct the full list preserving order, filling empty dicts for skipped items
        final_results = [{} for _ in texts]
        for idx, res in zip(valid_indices, results):
            final_results[idx] = res
            
        return final_results

    def load_from_sqlite(self, db_path: str, query: str = "SELECT * FROM reviews") -> pd.DataFrame:
        """
        Load reviews from a SQLite database into a pandas DataFrame.
        """
        try:
            print(f"Loading data from {db_path}...")
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"Loaded {len(df)} rows.")
            return df
        except Exception as e:
            print(f"Error loading from database: {e}")
            return pd.DataFrame()

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "review", batch_size: int = 32, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Process a DataFrame of reviews, adding classification scores as new columns.
        
        Args:
            df: The input DataFrame.
            text_column: The name of the column containing the review text.
            batch_size: Number of reviews to process at once. Defaults to 32.
            output_csv: Optional path to save the resulting DataFrame to CSV immediately.
        
        Returns:
            DataFrame with added score columns.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        print(f"Processing {len(df)} reviews with batch size {batch_size}...")
        
        # Ensure texts are strings and handle potential nulls
        texts = df[text_column].fillna("").astype(str).tolist()
        
        all_results = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
            batch_texts = texts[i : i + batch_size]
            batch_results = self.classify_batch(batch_texts)
            all_results.extend(batch_results)
            
        print("Formatting results into columns...")
        
        # Transform list of results into a DataFrame of scores
        score_data = []
        for res in all_results:
            row_scores = {}
            if res and 'labels' in res and 'scores' in res:
                for label, score in zip(res['labels'], res['scores']):
                    row_scores[f"score_{label}"] = score
            score_data.append(row_scores)
            
        scores_df = pd.DataFrame(score_data, index=df.index)
        
        # Concatenate original df with new scores
        result_df = pd.concat([df, scores_df], axis=1)
        
        if output_csv:
            print(f"Saving results to {output_csv}...")
            result_df.to_csv(output_csv, index=False)
            print("Save complete.")
            
        return result_df

    def print_result(self, text: str, result: Dict, threshold: float = 0.4):
        """
        Helper to print results nicely.
        """
        print(f"\nReview: {text[:100]}..." if len(text) > 100 else f"\nReview: {text}")
        print("-" * 50)
        
        labels = result.get('labels', [])
        scores = result.get('scores', [])
        
        for label, score in zip(labels, scores):
            if score >= threshold:
                print(f"  [{score:.4f}] {label}")
            else:
                pass

def main():
    # Example usage
    classifier = SteamZeroShotClassifier()

    # --- Database Mode (Example) ---
    # Uncomment the following lines to run on the full database
    db_path = "/content/gamesDB.db"
    # Adjust query to limit rows for testing if needed, e.g., "SELECT * FROM reviews LIMIT 100"
    query = "SELECT * FROM reviews" 
    
    # Load
    df = classifier.load_from_sqlite(db_path, query)
    
    if not df.empty:
        # Process and Export
        # Assuming the column with text is named 'review_text' or similar. Check your DB schema.
        result_df = classifier.process_dataframe(
            df, 
            text_column='review', # Change this to match your DB column name
            batch_size=64, 
            output_csv='classified_reviews.csv'
        )
        print("Head of processed data:")
        print(result_df.head())


if __name__ == "__main__":
    main()

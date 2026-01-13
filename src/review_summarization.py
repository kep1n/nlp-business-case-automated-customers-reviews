import sqlite3
import os
import json
import random
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from contextlib import contextmanager

import pandas as pd
from openai import OpenAI, OpenAIError, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SummarizerConfig:
    """Configuration for the Steam Review Summarizer."""
    sample_limit: int = 50
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_text_length: int = 8000
    min_review_length: int = 5
    min_reviews_required: int = 5
    summary_word_count: int = 100
    max_retries: int = 3
    retry_delay: int = 2  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class SummaryCache:
    """Manages caching of generated summaries."""

    def __init__(self, cache_dir: str = 'summaries_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'summaries.json'
        self.metadata_file = self.cache_dir / 'metadata.json'
        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.summaries = json.load(f)
        else:
            self.summaries = {}

        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'total_api_calls': 0,
                'total_tokens_used': 0,
                'estimated_cost': 0.0,
                'processed_games': []
            }

    def save_cache(self) -> None:
        """Persist cache to disk."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.summaries, f, indent=2, ensure_ascii=False)

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def get_summary(self, appid: int) -> Optional[str]:
        """Retrieve cached summary for a game."""
        return self.summaries.get(str(appid))

    def set_summary(self, appid: int, summary: str, tokens_used: int = 0) -> None:
        """Store summary in cache."""
        self.summaries[str(appid)] = summary

        # Update metadata
        self.metadata['total_api_calls'] += 1
        self.metadata['total_tokens_used'] += tokens_used
        # Rough estimate: GPT-3.5-turbo is ~$0.002 per 1K tokens
        self.metadata['estimated_cost'] = (self.metadata['total_tokens_used'] / 1000) * 0.002

        if str(appid) not in self.metadata['processed_games']:
            self.metadata['processed_games'].append(str(appid))

        self.save_cache()

    def is_processed(self, appid: int) -> bool:
        """Check if game has been processed."""
        return str(appid) in self.summaries

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_games_processed': len(self.summaries),
            'total_api_calls': self.metadata['total_api_calls'],
            'total_tokens_used': self.metadata['total_tokens_used'],
            'estimated_cost_usd': round(self.metadata['estimated_cost'], 4)
        }


class SteamReviewSummarizer:
    """
    Enhanced Steam review summarizer with caching, error handling, and persistence.
    """

    def __init__(
        self,
        db_path: str = 'gamesDB.db',
        api_key: Optional[str] = None,
        config: Optional[SummarizerConfig] = None
    ):
        """
        Initialize the summarizer with database path and OpenAI API key.

        Args:
            db_path (str): Path to the SQLite database.
            api_key (Optional[str]): OpenAI API key. If None, tries to fetch from environment.
            config (Optional[SummarizerConfig]): Configuration object. Uses defaults if None.
        """
        self.db_path = db_path
        self.config = config or SummarizerConfig()
        self.client = OpenAI(api_key=api_key)
        self.cache = SummaryCache()

        logger.info(f"Initialized SteamReviewSummarizer with config: {self.config.to_dict()}")

    @contextmanager
    def _get_db_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_reviews_by_game(self, review_type) -> Optional[pd.DataFrame]:
        """
        Reads reviews from the SQLite database using connection pooling.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing reviews or None if error/empty.
        """
        if not os.path.exists(self.db_path):
            logger.error(f"Database file '{self.db_path}' not found.")
            return None

        query = "SELECT reviews.appid, reviews.review, reviews.voted_up, games.review_score_desc, games.title FROM reviews JOIN games on reviews.appid = games.appid WHERE games.review_score_desc = '{}';".format(review_type)

        try:
            with self._get_db_connection() as conn:
                df = pd.read_sql_query(query, conn)
                logger.info(f"Loaded {len(df)} reviews from database.")
                return df
        except Exception as e:
            logger.error(f"Error reading database: {e}")
            return None

    def _sample_reviews(self, reviews: List[str], sample_size: int) -> List[str]:
        """
        Smart sampling with randomization to avoid chronological bias.

        Args:
            reviews (List[str]): List of reviews to sample from.
            sample_size (int): Desired sample size.

        Returns:
            List[str]: Sampled reviews.
        """
        if len(reviews) <= sample_size:
            return reviews
        return random.sample(reviews, sample_size)

    def _validate_summary(self, summary: str) -> bool:
        """
        Validate that summary contains all required sections.

        Args:
            summary (str): Generated summary text.

        Returns:
            bool: True if valid, False otherwise.
        """
        required_sections = [
            "Highlights",  # Flexible matching for "Top 3 Highlights"
            "Pain Points",  # Flexible matching for "Critical Pain Points"
            "Verdict"
        ]

        for section in required_sections:
            if section.lower() not in summary.lower():
                logger.warning(f"Summary missing section: {section}")
                return False

        return True

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count (4 chars ≈ 1 token).

        Args:
            text (str): Input text.

        Returns:
            int: Estimated token count.
        """
        return len(text) // 4

    def generate_summary(
        self,
        appid: int,
        game_reviews: pd.DataFrame,
        use_cache: bool = True
    ) -> Tuple[str, int]:
        """
        Generates a summary for a specific game using OpenAI API with retry logic.

        Args:
            appid (int): The application ID of the game.
            game_reviews (pd.DataFrame): DataFrame containing reviews for the specific game.
            use_cache (bool): Whether to use cached results.

        Returns:
            Tuple[str, int]: The generated summary and tokens used (or error message, 0).
        """
        # Check cache first
        if use_cache and self.cache.is_processed(appid):
            logger.info(f"Using cached summary for AppID {appid}")
            cached = self.cache.get_summary(appid)
            return cached, 0

        # Separate positive and negative reviews
        pos_reviews: List[str] = game_reviews[game_reviews['voted_up'] == 1]['review'].tolist()
        neg_reviews: List[str] = game_reviews[game_reviews['voted_up'] == 0]['review'].tolist()

        # Smart sampling with randomization
        pos_sample = self._sample_reviews(pos_reviews, self.config.sample_limit)
        neg_sample = self._sample_reviews(neg_reviews, self.config.sample_limit)

        # Filter very short/empty reviews
        pos_text = "\n- ".join([
            str(r) for r in pos_sample
            if r and len(str(r)) > self.config.min_review_length
        ])
        neg_text = "\n- ".join([
            str(r) for r in neg_sample
            if r and len(str(r)) > self.config.min_review_length
        ])

        # Truncate to avoid token limits
        pos_text = pos_text[:self.config.max_text_length]
        neg_text = neg_text[:self.config.max_text_length]

        system_prompt = "You are an expert video game journalist and data analyst specializing in horror/survival horror games."

        user_prompt = f"""
Analyze the following user reviews for the game (AppID: {appid}).

### Positive Reviews Sample:
{pos_text}

### Negative Reviews Sample:
{neg_text}

Based on these reviews, generate a {self.config.summary_word_count} words or less report containing:
1. **Top 3 Highlights**: The key reasons players recommend this game.
2. **Critical Pain Points**: The most recurring complaints or technical issues.
3. **The Verdict**: A summary sentence stating if the game is a "Must-Play", "Wait for Sale", or "Skip".

CONSTRAINTS:
    * Generate a smooth output text avoiding sections names esplicitly (Top 3 Highlights, Critical Pain Points and The Verdict) so it feels like natural language. Take into account that the output will be posted on a gamers blog.
    * Do not make up for features that are not reflected in the text. As a journalist you are giving your an opinion (subjective) but based on the reviews (facts).
"""

        # Retry logic with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1} for AppID {appid}")

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                )

                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0

                if not content:
                    logger.error("Empty response from API.")
                    return "Error: Empty response from API.", 0

                # Validate summary
                if not self._validate_summary(content):
                    logger.warning(f"Summary for AppID {appid} failed validation but returning anyway.")

                # Cache the result
                self.cache.set_summary(appid, content, tokens_used)

                logger.info(f"Successfully generated summary for AppID {appid} (Tokens: {tokens_used})")
                return content, tokens_used

            except RateLimitError as e:
                logger.warning(f"Rate limit hit for AppID {appid}: {e}")
                if attempt < self.config.max_retries - 1:
                    sleep_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    error_msg = f"Rate limit error after {self.config.max_retries} attempts: {e}"
                    logger.error(error_msg)
                    return error_msg, 0

            except OpenAIError as e:
                logger.error(f"OpenAI API error for AppID {appid}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return f"OpenAI API error: {e}", 0

            except Exception as e:
                logger.error(f"Unexpected error generating summary for AppID {appid}: {e}")
                return f"Error generating summary: {e}", 0

        return "Error: Maximum retries exceeded.", 0

    def get_review_statistics(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get statistics about the review dataset.

        Args:
            df (pd.DataFrame): Reviews dataframe.

        Returns:
            Dict[str, int]: Statistics dictionary.
        """
        return {
            'total_reviews': len(df),
            'positive_reviews': len(df[df['voted_up'] == True]),
            'negative_reviews': len(df[df['voted_up'] == False]),
            'unique_games': df['appid'].nunique()
        }

    def export_summaries_to_json(self, output_path: str = 'summaries_export.json') -> None:
        """
        Export all cached summaries to a JSON file for web app integration.

        Args:
            output_path (str): Path to output JSON file.
        """
        export_data = {
            'summaries': self.cache.summaries,
            'metadata': self.cache.metadata,
            'config': self.config.to_dict()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported summaries to {output_path}")

    def run(
        self,
        limit: Optional[int] = 3,
        skip_processed: bool = True,
        export_results: bool = True
    ) -> None:
        """
        Main execution method to process games with progress tracking.

        Args:
            limit (Optional[int]): Number of games to process. None for all.
            skip_processed (bool): Skip games that are already cached.
            export_results (bool): Export results to JSON after completion.
        """
        logger.info(f"Loading reviews from {self.db_path}...")
        review_type = 'Mostly Negative'
        df = self.get_reviews_by_game(review_type)

        if df is None or df.empty:
            logger.error("No reviews found or could not load database.")
            return

        # Show statistics
        stats = self.get_review_statistics(df)
        logger.info(f"Dataset statistics: {stats}")

        unique_games = df['appid'].unique()
        logger.info(f"Found {len(unique_games)} unique games.")

        # Filter games to process
        if skip_processed:
            games_to_process = [
                appid for appid in unique_games
                if not self.cache.is_processed(appid)
            ]
            logger.info(f"Skipping {len(unique_games) - len(games_to_process)} already processed games.")
        else:
            games_to_process = unique_games

        # Apply limit
        if limit:
            games_to_process = games_to_process[:limit]

        logger.info(f"Processing {len(games_to_process)} games...")

        # Process with progress bar
        for appid in tqdm(games_to_process, desc="Processing games"):
            logger.info(f"Processing Game AppID: {appid}")
            game_reviews = df[df['appid'] == appid]
            title = game_reviews.iloc[0, game_reviews.columns.get_loc('title')]

            # Skip if too few reviews for a meaningful summary
            if len(game_reviews) < self.config.min_reviews_required:
                logger.warning(f"Skipping {appid} (only {len(game_reviews)} reviews)")
                continue

            summary, tokens = self.generate_summary(int(appid), game_reviews)
            with open(f'{os.path.dirname(__file__)}\\{appid}_{title}_{review_type}.txt', 'w', encoding='utf-8') as summary_txt:
                summary_txt.write(f"SUMMARY FOR GAME {title}\n")
                summary_txt.write("\n")
                summary_txt.write(summary)

        # Show final statistics
        cache_stats = self.cache.get_statistics()
        logger.info(f"Processing complete. Cache statistics: {cache_stats}")

        # Export results
        if export_results:
            self.export_summaries_to_json()


def main():
    """Main execution function with custom configuration."""
    # Custom configuration (optional)
    config = SummarizerConfig(
        sample_limit=50,
        model="gpt-4o-mini",
        temperature=0.7,
        summary_word_count=500,
        max_retries=3
    )

    summarizer = SteamReviewSummarizer(
        db_path=r'E:\SynologyDrive\ironhack\week_14\day_2\nlp-business-case-automated-customers-reviews\data\raw\gamesDB.db',
        config=config
    )

    # Process games with caching and export
    summarizer.run(
        limit=5,
        skip_processed=True,
        export_results=True
    )

    # Show final statistics
    stats = summarizer.cache.get_statistics()
    logger.info(f"Final statistics: {stats}")


if __name__ == "__main__":
    main()

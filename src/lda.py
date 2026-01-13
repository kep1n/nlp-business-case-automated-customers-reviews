from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import sqlite3
import mlflow
import pandas as pd
from nltk.corpus import stopwords
import re
import os
import pickle
import joblib
# from google.colab import userdata
from dotenv import load_dotenv
import nltk
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()

def load_data(db_path, table_name):
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query: Select only the text column you need
    # Make sure to handle NULLs to avoid crashing the vectorizer
    query = f"SELECT review, recommendationid FROM {table_name}"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df)} reviews.")
    return df

def clean_steam_review(text: str) -> str:
    """Minimal cleaning preserving context for BERT"""

    if not isinstance(text, str) or not text.strip():
        return ""
    
    STEAM_PATTERNS = [
        r"\b\d{1,4}\s?hrs?\b",
        r"\b\d{1,4}\s?hours?\b",
        r"\bplay(ed|ing)?\b",
        r"\bupdate\b",
        r"\bpatch\b",
        r"\bdevs?\b",
        r"\bdev(eloper)?s?\b",
        r"\bearly access\b",
        r"\brefund(ed|ing)?\b",
        r"\brecommend(ed|ing)?\b",
        r"\bnot recommend(ed|ing)?\b",
        r"\bworth it\b",
        r"\bprice\b",
        r"\bperformance\b",
        r"\bfps\b",
        r"\blag\b",
        r"\bcrash(es|ed)?\b",
    ]
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        tagged = pos_tag(words)

        lemmas = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in tagged
        ]
        return " ".join(lemmas)

    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"http\S+", " ", text)   # remove URLs
        text = re.sub(r"[^a-z\s]", " ", text)  # keep letters only
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    for p in STEAM_PATTERNS:
        text = re.sub(p, " ", text, flags=re.I)
    
    text = lemmatize_text(text)

    return text

def get_top_words(lda, feature_names, n_top_words=10):
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features = topic.argsort()[:-n_top_words - 1:-1]
        topics[topic_idx] = [feature_names[i] for i in top_features]
    return topics

def main():
    DB_PATH = r'E:\SynologyDrive\ironhack\week_14\day_2\nlp-business-case-automated-customers-reviews\data\raw\gamesDB.db'
    TABLE_NAME = 'reviews'

    df = load_data(DB_PATH, TABLE_NAME)
    # Vectorize
    custom_stopwords = [
        'game', 'play', 'horror', 'scary', 'recommend', 'best', 'bad'
        'worth', 'fun', 'good', 'bad', 'great', 'get', 'like',
        'time', 'youtube', 'love', 'hate', 'awesome', 'terrible', 'awful', 'recommend',
        'steam', 'steampowered', 'alan', 'wake', 'silent', 'hill', 'redient', 'evil', 'dead'
    ]

    vectorizer = CountVectorizer(
        max_features=1500,
        stop_words=list(set(stopwords.words('english') + custom_stopwords)),
        ngram_range=(1, 2),
        min_df=15,
        max_df=0.6  # Ignore words in >60% of reviews
    )

    df = df[df['review'].str.len() >= 20]

    df['clean_text'] = df['review'].apply(clean_steam_review)

    X = vectorizer.fit_transform(df['clean_text'])

    # LDA
    lda = LatentDirichletAllocation(
        n_components=3,  # Fewer topics or 7
        random_state=42,
        n_jobs=1
    )
    topics = lda.fit_transform(X)

    feature_names = vectorizer.get_feature_names_out()
    topic_words = get_top_words(lda, feature_names)

    for k, words in topic_words.items():
        print(f"Topic {k}: {', '.join(words)}")
    perplexity = lda.perplexity(X)
    print("Perplexity:", perplexity)

    df["topic"] = topics.argmax(axis=1)
    df.to_csv('lda_topics.csv', index=False, sep=';')

    joblib.dump(
    {
        "vectorizer": vectorizer,
        "lda": lda
    },
    "lda_pipeline.joblib"
)

if __name__ == '__main__':
    mlflow.set_experiment('steam-reviews-clustering')
    mlflow.autolog()
    with mlflow.start_run():
        main()
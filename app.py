import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
from transformers import pipeline
from sklearn.cluster import KMeans
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import io
import os
# Set NLTK data path for both local and cloud
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
nltk.data.path.insert(0, NLTK_DATA_PATH)

def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1], download_dir=NLTK_DATA_PATH)

safe_nltk_download('tokenizers/punkt')
safe_nltk_download('corpora/stopwords')

# Load models
@st.cache_resource
def load_use_encoder():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

USE_encoder = load_use_encoder()
sentiment_model = load_sentiment_model()

# Helper functions
def embed(input):
    return np.array(USE_encoder(input))

def sentiment(input):
    result = sentiment_model(input[:512])
    sign = 1 if(result[0]['label']=="POSITIVE") else -1
    value = result[0]['score']
    return sign*value

class ReviewsTopicModel:
    STOPWORDS = stopwords.words('english')
    EMBEDDING_DIM = 512
    def __init__(self, reviews, ids):
        self.review_texts = reviews  # Save original review texts
        self.ids = ids  # Save IDs
        self.X = self.clean(reviews)
        self.X.index = ids  # Keep IDs as index for merging
        self.id_to_review = dict(zip(ids, reviews))  # Map ID to review text
    def clean(self, reviews):
        string_map = {'\r': '', '\n': '', '/': ' ', "'": "", '"': ''}
        reviews_cleaned = reviews[:]
        for i in range(len(reviews_cleaned)):
            for s in string_map:
                reviews_cleaned[i] = reviews_cleaned[i].replace(s, string_map[s]).lower()
        X = embed(reviews_cleaned)
        X = pd.DataFrame(X)
        X.index = reviews_cleaned
        return X
    def elbow_plot(self):
        if len(self.X) < 2:
            st.warning("Need at least 2 samples to plot the elbow curve.")
            return
        max_clusters = min(80, len(self.X))
        cluster_sizes = list(range(1, max_clusters + 1))
        cluster_scores = []
        for n in cluster_sizes:
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(self.X)
            cluster_scores.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        ax.plot(cluster_sizes, cluster_scores)
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Plot')
        st.pyplot(fig)
    def create_topics(self, num_topics):
        kmeans = KMeans(n_clusters=num_topics)
        kmeans.fit(self.X)
        topics_df = self.X.copy()
        topics_df['topic'] = kmeans.labels_.copy()
        topics_df['topic'] = topics_df['topic'].astype(int)
        topic_keywords = {}
        for topic in topics_df['topic'].unique():
            topic_ids = topics_df.query(f"topic == {topic}").index.tolist()
            # Use original review text for keyword extraction
            topic_reviews = [self.id_to_review[_id] for _id in topic_ids]
            topic_centroid = kmeans.cluster_centers_[topic]
            topic_keywords[topic] = self.get_closest_words(topic_reviews, topic_centroid)
        topics_df['topic_keywords'] = topics_df['topic'].map(topic_keywords)
        topics_df['sentiment'] = [sentiment(self.id_to_review[_id]) for _id in topics_df.index.values.tolist()]
        self.topics_keywords = topic_keywords
        self.topics_df = topics_df.copy()[['topic', 'topic_keywords', 'sentiment']]
    def get_closest_words(self, reviews, centroid):
        word_distances = {}
        for r in reviews:
            review_words = [w for w in word_tokenize(r) if(w not in self.STOPWORDS)]
            for w in review_words:
                word_embedding = embed([w])
                word_distances[w] = self.cosine_similarity(word_embedding, centroid)
        top_5_keywords = sorted([(word_distances[w], w) for w in word_distances])[-5:]
        return ",".join([x[1] for x in top_5_keywords])
    def cosine_similarity(self, x, y):
        x = x.reshape(self.EMBEDDING_DIM,)
        y = y.reshape(self.EMBEDDING_DIM,)
        dotproduct = x.dot(y)
        x_mag = x.dot(x)**0.5
        y_mag = y.dot(y)**0.5
        return dotproduct/(x_mag * y_mag)

# Streamlit UI
st.title("Robs Wonderful Topic Modeling Site")
st.write("Upload a .tsv file with a 'Transcript' column to analyze topics and sentiment.")

uploaded_file = st.file_uploader("Choose a .tsv file", type=["tsv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep="\t")
    st.write(f"Data shape: {df.shape}")
    st.write(df.head())
    if 'Transcript' not in df.columns or 'ID' not in df.columns:
        st.error("No 'Transcript' or 'ID' column found in the uploaded file.")
    else:
        # Filter out transcripts that only say 'Not Available'
        reviews_df = df[['ID', 'Transcript']].dropna()
        reviews_df = reviews_df[reviews_df['Transcript'].str.strip().str.lower() != 'not available']
        reviews = reviews_df['Transcript'].tolist()
        review_ids = reviews_df['ID'].tolist()
        if len(reviews) > 500:
            st.warning("Large files may cause memory errors on Streamlit Cloud. Try a smaller file or fewer reviews.")
        topic_model = ReviewsTopicModel(reviews, review_ids)
        st.subheader("Elbow Plot (for topic selection)")
        if len(reviews) < 2:
            st.warning("Need at least 2 samples to plot the elbow curve.")
        else:
            topic_model.elbow_plot()
        num_topics = st.number_input("Number of topics", min_value=2, max_value=80, value=5)
        @st.cache_data(show_spinner=False)
        def get_topic_model_results(reviews, review_ids, num_topics):
            model = ReviewsTopicModel(reviews, review_ids)
            model.create_topics(num_topics=num_topics)
            return model.topics_df.copy(), model.topics_keywords.copy()
        if st.button("Run Topic Modeling"):
            with st.spinner('Running topic modeling...'):
                topics_df, topics_keywords = get_topic_model_results(reviews, review_ids, num_topics)
                st.session_state['topics_df'] = topics_df
                st.session_state['topics_keywords'] = topics_keywords
                st.session_state['num_topics'] = num_topics
                st.success("Topic modeling complete!")
        if 'topics_df' in st.session_state and st.session_state.get('num_topics') == num_topics:
            st.subheader("Topic Keywords and Sentiment")
            # Fix: Show only the keywords, not the index, for each topic
            result_df = st.session_state['topics_df'].groupby(['topic']).agg({'topic_keywords': 'first', 'sentiment': 'mean'}).reset_index().sort_values(by='sentiment')
            st.dataframe(result_df)
            st.subheader("Explore Reviews by Topic")
            topic_choice = st.selectbox("Select topic to view reviews", result_df['topic'])
            reviews_for_topic_df = st.session_state['topics_df'][st.session_state['topics_df']['topic'] == topic_choice].copy()
            merged_df = reviews_for_topic_df.merge(df[['ID', 'Transcript', 'YouTube URL']], left_index=True, right_on='ID', how='left')
            merged_df.rename(columns={'YouTube URL': 'URL'}, inplace=True)
            st.write(merged_df[['Transcript', 'sentiment', 'URL']])

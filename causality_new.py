'''This code file helps to pool domain hotwords into clusters and trends based on statistical
and algebraical measures
Input: News data, stock data
Output: Causal clusters, causal strengths between clusters and trends'''
#!pip install pandas numpy scikit-learn scipy yake nltk gensim causal-learn
"""Necessary Packages"""
import pandas as pd
import numpy as np
import yake
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from itertools import combinations
from gensim.models import Word2Vec
from scipy.stats import ks_2samp
import networkx as nx
import plotly.graph_objs as go
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

"""Function for data pre-processing:
The news dataframe and stock dataframe is merged based on matching dates, with two additional features: Return and Trend.
Missing values are either filled based on previous records, or default to Neutral trend.
The news dataset was a pre-curated one from Nifty50, containing 2718 instances from January 2020 to April 2024. The stock price
data from the same dates were extracted from YahooFinance."""
def load_and_preprocess_data(news_file, stock_file):
    print('Preprocessing...')
    news_df = pd.read_csv(news_file)
    stock_df = pd.read_csv(stock_file)

    news_df['datePublished'] = pd.to_datetime(news_df['datePublished'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    news_df = news_df.sort_values('datePublished')
    stock_df = stock_df.sort_values('Date')

    stock_df['Return'] = stock_df['Close'].pct_change()

    stock_df['Trend'] = pd.cut(stock_df['Return'],
                               bins=[-np.inf, -0.001, 0.001, np.inf],
                               labels=['Down', 'Neutral', 'Up'])

    news_df['Date'] = news_df['datePublished'].dt.date
    stock_df['Date'] = stock_df['Date'].dt.date
    merged_df = pd.merge(news_df, stock_df[['Date', 'Trend']], on='Date', how='left')

    merged_df['Trend'] = merged_df['Trend'].ffill()
    merged_df['Trend'] = merged_df['Trend'].fillna('Neutral')

    return merged_df, stock_df

merged_df, _ = load_and_preprocess_data('News_Train.csv', 'Nifty_50_data.csv')
merged_df.to_csv('merged_data.csv')
print('Merge saved')

"""Extraction of keywords from financial news"""
def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=20, features=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

"""Creation of trend corpora: A corpus for each financial trend is created by merging together the texts belonging
to each trend."""
def create_trend_corpora(merged_df):
    trend_corpora = {}
    for trend in merged_df['Trend'].unique():
        trend_articles = merged_df[merged_df['Trend'] == trend]['articleBody']
        trend_corpora[trend] = ' '.join(trend_articles)
    return trend_corpora

"""This function calculates the similarity between each cluster formed and the trend corpus belonging to each trend using
TF-IDF to see which cluster matches which trend the most."""
def calculate_cluster_trend_similarities(causal_factor_clusters, trend_corpora):
    # Create a corpus for each cluster
    cluster_corpora = {i: ' '.join(words) for i, words in causal_factor_clusters.items()}

    # Combine cluster and trend corpora
    all_corpora = {**cluster_corpora, **trend_corpora}

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_corpora.values())

    # Calculate similarities
    similarities = cosine_similarity(tfidf_matrix)

    return similarities, list(all_corpora.keys())

"""Domain hotwords are extracted from every piece of financial news in the following way:
1. The text is tokenized into words.
2. Words receive POS labels.
3. Words with verb (VB) or adverb (RB) tagging get selected as domain hotwords.
4. 15 most frequent hotwords in each trend are selected as the primary hotwords for that trend."""
def extract_domain_hotwords(merged_df):
    print('Hotword extraction...')
    stop_words = set(stopwords.words('english'))
    hotwords = {}

    for trend, group in merged_df.groupby('Trend'):
        all_words = []
        for text in group['articleBody']:
            tokens = word_tokenize(text.lower())
            tagged = nltk.pos_tag(tokens)
            words = [word for word, tag in tagged
                     if (tag.startswith('VB') or tag.startswith('RB'))
                     and word not in stop_words]
            all_words.extend(words)

        word_freq = nltk.FreqDist(all_words)
        hotwords[trend] = [word for word, _ in word_freq.most_common(15)]

    return hotwords

"""This function checks if a particular word in a piece of news affects its trend enough to distinguish it from another
piece of news. For example, if the frequency of a word x is high in article A and low in Article B, while article A has 
a positive trend and Article B has a negative trend, it can be inferred that x is a distinguishable keyword."""
def distinguishable_word_filtering(word, group_i, group_j, alpha=0.05):
    freq_i = [text.count(word) for text in group_i['articleBody']] #Frequency of an word in article i
    freq_j = [text.count(word) for text in group_j['articleBody']] # Frequency of a word in article j

    # Check if either frequency list is empty
    if not freq_i or not freq_j:
        return False  # Consider words that do not occur in one of the groups as indistinguishable

    _, p_value = ks_2samp(freq_i, freq_j) #Function for K-S testing (paper linked in report)
    return p_value <= alpha

"""Using the function distinguishable_word_filtering, distinguishable keywords are extracted from the entire merged dataframe."""
def extract_distinguishable_words(merged_df):
    print('Distinguishable...')
    trends = merged_df['Trend'].unique()
    distinguishable_words = set()

    for word in set(' '.join(merged_df['articleBody']).split()):
        distinguishable_count = 0
        total_comparisons = 0

        for i, trend_i in enumerate(trends):
            for j, trend_j in enumerate(trends):
                if i < j:
                    group_i = merged_df[merged_df['Trend'] == trend_i]
                    group_j = merged_df[merged_df['Trend'] == trend_j]

                    if group_i.empty or group_j.empty:
                        print(f"Skipping comparison for word '{word}' between trends '{trend_i}' and '{trend_j}' due to empty group.")
                        continue

                    if distinguishable_word_filtering(word, group_i, group_j):
                        distinguishable_count += 1
                    total_comparisons += 1

        if distinguishable_count > total_comparisons / 2:
            distinguishable_words.add(word)

    return list(distinguishable_words)

"""This function clusters all the extracted keywords into 50 clusters. The keywords are vectorized into Word2vec embeddings,
and based on the similarity between these keywords, they are clustered using K-means clustering."""
def cluster_keywords(keywords, n_clusters=50):
    print('Clustering...')
    sentences = [keywords]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    word_vectors = [model.wv[word] for word in keywords if word in model.wv]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(word_vectors)

    clustered_keywords = {i: [] for i in range(n_clusters)}
    for keyword, cluster in zip(keywords, clusters):
        clustered_keywords[cluster].append(keyword)

    return clustered_keywords

"""This function, convert_to_numerical_data, converts text data in merged_df
to a numerical format suitable for machine learning by mapping specific words in each article
to "clusters" (groups defined in causal_factors) and encoding trend labels.
1. count_clusters is a nested function that takes in text (article body), 
splits it into words, and counts how often words from each cluster appear.
2. It processes all articles in merged_df['articleBody'] and builds a matrix X with one row per article 
and one column per cluster, where each element is the word count for that cluster in the article.
3. The code then uses one-hot encoding to convert the trend labels in merged_df['Trend'] into numerical format."""
def convert_to_numerical_data(merged_df, causal_factors):
    word_to_cluster = {}
    for cluster, words in causal_factors.items():
        for word in words:
            word_to_cluster[word] = cluster

    def count_clusters(text):
        words = text.split()
        cluster_counts = Counter(word_to_cluster.get(word, -1) for word in words if word in word_to_cluster)
        return [cluster_counts.get(i, 0) for i in range(len(causal_factors))]

    X = np.array(merged_df['articleBody'].apply(count_clusters).tolist())

    # One-hot encode the trend
    trend_encoding = pd.get_dummies(merged_df['Trend'], prefix='Trend')
    X = np.hstack([X, trend_encoding.values])

    feature_names = [f"Cluster_{i}" for i in range(len(causal_factors))] + list(trend_encoding.columns)

    return X, feature_names

def create_causal_graph_1(similarities, corpus_keys, causal_factor_clusters):
    num_clusters = len(causal_factor_clusters)
    num_trends = len(corpus_keys) - num_clusters

    causal_strengths = {}

    for i in range(num_clusters):
        cluster_similarities = similarities[i, num_clusters:]
        max_similarity_index = np.argmax(np.abs(cluster_similarities))
        max_similarity = cluster_similarities[max_similarity_index]

        trend = corpus_keys[num_clusters + max_similarity_index]
        causal_strengths[(i, f"Trend_{trend}")] = max_similarity

    return causal_strengths

"""This function constructs a graph using NetworkX package.
The nodes consist of the 50 clusters and 3 trend nodes (representing up, neutral, down).
The edges consist of the causal strengths between the nodes (clusters and trends)."""
def visualize_causal_graph_interactive(causal_strengths, feature_names, save_as_html=False, file_name="new_causal_graph.html", threshold=0.000):
    G = nx.DiGraph()

    # Add nodes with labels
    for idx, name in enumerate(feature_names):
        G.add_node(idx, label=name)

    # Add edges
    for edge, strength in causal_strengths.items():
        #if abs(strength) <= threshold:
        #    continue
        source = edge[0] if isinstance(edge[0], int) else feature_names.index(edge[0])
        target = edge[1] if isinstance(edge[1], int) else feature_names.index(edge[1])
        G.add_edge(source, target, weight=strength)

    pos = nx.spring_layout(G, seed=42)

    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=2, color='rgba(0, 0, 0, 0.5)'),
                hoverinfo='text',
                text=f"{feature_names[edge[0]]} -> {feature_names[edge[1]]}<br>Strength: {weight:.4f}",
                mode='lines'
            )
        )

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([G.nodes[node]['label']])

    fig = go.Figure(data=[*edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>New Causal Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    if save_as_html:
        fig.write_html(file_name)
    fig.show()

"""Driver code"""
if __name__ == "__main__":
    news_file = "News_Train.csv"
    stock_file = "Nifty_50_data.csv"

    merged_df, stock_df = load_and_preprocess_data(news_file, stock_file)
    #merged_df = merged_df[:20]

    # Extract keywords
    merged_df['keywords'] = merged_df['articleBody'].apply(extract_keywords)
    all_keywords = [word for keywords in merged_df['keywords'] for word in keywords]

    # Extract domain hotwords and distinguishable words
    hotwords = extract_domain_hotwords(merged_df)
    distinguishable_words = extract_distinguishable_words(merged_df)

    # Combine all keywords
    all_causal_factors = list(set(all_keywords + [word for trend_words in hotwords.values() for word in trend_words] + distinguishable_words))

    # Cluster keywords
    causal_factor_clusters = cluster_keywords(all_causal_factors)

    # Convert to numerical data
    data, feature_names = convert_to_numerical_data(merged_df, causal_factor_clusters)

    # Create trend corpora
    trend_corpora = create_trend_corpora(merged_df)

    # Calculate similarities
    similarities, corpus_keys = calculate_cluster_trend_similarities(causal_factor_clusters, trend_corpora)

    # Create causal graph
    causal_strengths = create_causal_graph_1(similarities, corpus_keys, causal_factor_clusters)

    combined_strengths = {}
    #combined_strengths.update(strengths)
    combined_strengths.update(causal_strengths)

    # Print results
    print("Causal Factor Clusters:")
    for i, cluster in enumerate(causal_factor_clusters.values()):
        print(f"Cluster {i}: {cluster}")

    clusters_dict = {f"Cluster_{i}": cluster for i, cluster in enumerate(causal_factor_clusters.values())}

    # Write to JSON file
    with open('clusters_new.json', 'w', encoding='utf-8') as f:
        json.dump(clusters_dict, f, ensure_ascii=False, indent=4)

    print("Clusters saved to clusters.json")

    print("\nCausal Graph Edges:")
    for edge, strength in combined_strengths.items():
        if type(edge[1]) == str:
            print(f"{feature_names[edge[0]]} -> {edge[1]}: Strength = {strength:.4f}")
            continue
        print(f"{feature_names[edge[0]]} -> {feature_names[edge[1]]}: Strength = {strength:.4f}")

    json_strengths = {}

    for edge, strength in combined_strengths.items():
        # Convert tuple of indices to string of feature names
        if type(edge[1]) == str:
            key = f"{feature_names[edge[0]]} -> {edge[1]}"
            json_strengths[key] = strength
            continue
        key = f"{feature_names[edge[0]]} -> {feature_names[edge[1]]}"
        json_strengths[key] = strength

    # Write to JSON file
    with open('combined_strengths.json', 'w') as f:
        json.dump(json_strengths, f, indent=4)

    print("Combined strengths saved to combined_strengths.json")

    # Visualize causal graph
    visualize_causal_graph_interactive(combined_strengths, feature_names, save_as_html=True, threshold = 0.000)


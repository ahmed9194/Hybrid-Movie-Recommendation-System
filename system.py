import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import requests
from io import BytesIO

@st.cache_data
def load_data():
    # Load data with data cleaning
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    
    # Data cleaning section
    with st.spinner('Cleaning data...'):
        # Handle missing values
        initial_count = len(movies)
        movies = movies.dropna(subset=['title', 'genres'])
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        
        # Extract and validate year
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies = movies[movies['year'].notna()]
        movies['year'] = movies['year'].astype(int)
        movies = movies[movies['year'].between(1900, 2025)]
        
        # Remove duplicate ratings
        ratings = ratings.drop_duplicates(['userId', 'movieId'])
        
    # Show data cleaning report
    if 'show_clean_report' not in st.session_state:
        st.session_state.show_clean_report = True
        st.subheader("🧹 Data Cleaning Report")
        col1, col2 = st.columns(2)
        col1.metric("Movies After Cleaning", len(movies), delta=f"{-initial_count + len(movies)}")
        col2.metric("Valid Ratings", len(ratings))
        
    return movies, ratings

def get_movie_poster(title):
    """Generate placeholder movie poster based on title"""
    try:
        # Clean title for URL
        clean_title = title[:20].replace(" ", "+").replace("(", "").replace(")", "")
        return f"https://via.placeholder.com/150x225.png?text={clean_title}"
    except:
        return "https://via.placeholder.com/150x225.png?text=No+Poster"

def explore_data(movies, ratings):
    st.header("📊 Data Exploration")

    # Basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Movies", len(movies))
    col2.metric("Total Ratings", len(ratings))
    col3.metric("Unique Users", ratings['userId'].nunique())

    # Movies per year
    movies_per_year = movies['year'].value_counts().sort_index().reset_index()
    movies_per_year.columns = ['Year', 'Count']
    
    fig1 = px.line(movies_per_year, x='Year', y='Count', 
                  title='Movies Released Per Year')
    st.plotly_chart(fig1, use_container_width=True)

    # Ratings distribution
    fig2 = px.histogram(ratings, x='rating', nbins=10, 
                       title='Distribution of Ratings')
    st.plotly_chart(fig2, use_container_width=True)

    # Genre word cloud
    st.subheader("Genre Word Cloud")
    genres_text = ' '.join(movies['genres'].str.replace('|', ' '))
    wordcloud = WordCloud(width=800, height=400).generate(genres_text)
    fig3 = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig3)

def enhanced_content_based_recommender(movies, movie_title, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices.get(movie_title)
    if idx is None:
        return pd.DataFrame([])

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][['title', 'genres', 'year']].copy()
    recommendations['similarity'] = [f"{i[1]*100:.1f}%" for i in sim_scores]
    recommendations['poster'] = recommendations['title'].apply(lambda x: get_movie_poster(x))
    recommendations['score'] = [i[1] for i in sim_scores]  # Add raw score for hybrid
    recommendations['type'] = 'Content-Based'

    return recommendations

def collaborative_filtering_sklearn(ratings, movies, user_id, top_n=5):
    # Create user-movie matrix
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame([])
    
    # Train model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_movie_matrix)
    
    # Find similar users
    distances, indices = model.kneighbors([user_movie_matrix.loc[user_id]], n_neighbors=6)
    
    # Aggregate ratings from similar users
    similar_users = user_movie_matrix.iloc[indices[0][1:]]  # Skip the user themselves
    recommendations = similar_users.mean(axis=0).sort_values(ascending=False)[:top_n]
    
    # Prepare results
    result = []
    for movie_id, score in recommendations.items():
        try:
            movie_info = movies[movies['movieId'] == movie_id].iloc[0]
            result.append({
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'year': movie_info.get('year', 'N/A'),
                'predicted_rating': f"{score:.1f}/5.0",
                'poster': get_movie_poster(movie_info['title']),
                'score': score/5,  # Normalized score for hybrid
                'type': 'Collaborative'
            })
        except:
            continue
    
    return pd.DataFrame(result)

def hybrid_recommender_with_viz(movies, ratings, user_id, movie_title, top_n=5):
    # Get recommendations from both methods
    content_recs = enhanced_content_based_recommender(movies, movie_title, top_n*2)
    collab_recs = collaborative_filtering_sklearn(ratings, movies, user_id, top_n*2)
    
    # Combine with weights (adjust these based on preference)
    content_weight = 0.4
    collab_weight = 0.6
    
    if not content_recs.empty:
        content_recs['weighted_score'] = content_recs['score'] * content_weight
    
    if not collab_recs.empty:
        collab_recs['weighted_score'] = collab_recs['score'] * collab_weight
    
    # Combine and sort by weighted score
    combined = pd.concat([content_recs, collab_recs])
    combined = combined.sort_values('weighted_score', ascending=False).head(top_n)
    
    # Visualization
    st.subheader("Recommendation Composition")
    fig = px.pie(combined, names='type', title='Recommendation Sources')
    st.plotly_chart(fig)
    
    return combined

def main():
    st.title("🎬 Hybrid Movie Recommendation System")
    
    # Load data
    movies, ratings = load_data()
    
    # Navigation
    page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Get Recommendations"])
    
    if page == "Data Exploration":
        explore_data(movies, ratings)
    else:
        st.header("🔍 Personalized Recommendations")
        
        col1, col2 = st.columns(2)
        user_id = col1.number_input("Enter User ID:", 
                                  min_value=1, 
                                  max_value=ratings['userId'].max(), 
                                  value=1, 
                                  step=1)
        
        # Movie search with autocomplete
        movie_list = movies['title'].tolist()
        default_index = movie_list.index("Toy Story (1995)") if "Toy Story (1995)" in movie_list else 0
        movie_title = col2.selectbox("Select a Movie You Like:", 
                                   movie_list, 
                                   index=default_index)
        
        top_n = st.slider("Number of Recommendations:", 1, 20, 5)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner('Generating recommendations...'):
                recommendations = hybrid_recommender_with_viz(movies, ratings, user_id, movie_title, top_n)
                
                if recommendations.empty:
                    st.warning("No recommendations found. Try different inputs.")
                else:
                    st.success("Recommended Movies For You:")
                    for _, row in recommendations.iterrows():
                        st.markdown(f"### {row['title']} ({row.get('year', '')})")
                        cols = st.columns([1, 3])
                        cols[0].image(row['poster'], use_column_width=True)
                        
                        with cols[1]:
                            st.write(f"**Genres:** {row['genres']}")
                            if 'similarity' in row:
                                st.write(f"**Content Similarity:** {row['similarity']}")
                            if 'predicted_rating' in row:
                                st.write(f"**Predicted Rating:** {row['predicted_rating']}")
                            st.write(f"**Recommendation Type:** {row['type']}")
                            st.progress(float(row['weighted_score']))

if __name__ == "__main__":
    main()

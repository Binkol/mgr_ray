###### Step 1: Import Python Libraries

# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns

# Similarity
from sklearn.metrics.pairwise import cosine_similarity


###### Step 2: Download And Read In Data

# # Read in data
ratings=pd.read_csv('data/ml-latest-small/ratings.csv')

# # Take a look at the data
# #ratings.head()

# # Get the dataset information
# #ratings.info()

# # Number of users
# #print('The ratings dataset has', ratings['userId'].nunique(), 'unique users')

# # Number of movies
# #print('The ratings dataset has', ratings['movieId'].nunique(), 'unique movies')

# # Number of ratings
# #print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings')

# # List of unique ratings
# #print('The unique ratings are', sorted(ratings['rating'].unique()))

# # Read in data
movies = pd.read_csv('data/ml-latest-small/movies.csv')

# # Take a look at the data
# #movies.head()

# # Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='movieId', how='inner')

# # Take a look at the data
# #df.head()


# ###### Step 3: Exploratory Data Analysis (EDA)

# # Aggregate by movie
agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()

# # Keep the movies with over X ratings
agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>30]
#print(agg_ratings_GT100.head())
#agg_ratings_GT100.info()

# # Check popular movies
#agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()

# # Visulization
# #sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)

# # Merge data
df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
# #df_GT100.info()

# # Number of users
# #print('The ratings dataset has', df_GT100['userId'].nunique(), 'unique users')

# # Number of movies
# #print('The ratings dataset has', df_GT100['movieId'].nunique(), 'unique movies')

# # Number of ratings
# #print('The ratings dataset has', df_GT100['rating'].nunique(), 'unique ratings')

# # List of unique ratings
# #print('The unique ratings are', sorted(df_GT100['rating'].unique()))


# ###### Step 4: Create User-Movie Matrix

# # Create user-item matrix
matrix = df_GT100.pivot_table(index='title', columns='userId', values='rating')
# #matrix.head()


# ###### Step 5: Data Normalization

# # Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
#print(matrix_norm.head())


# ###### Step 6: Calculate Similarity Score

# # Item similarity matrix using Pearson correlation
item_similarity = matrix_norm.T.corr()
# #item_similarity.head()

# # Item similarity matrix using cosine similarity
# item_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
# #item_similarity_cosine


# ###### Step 7: Predict User's Rating For One Movie

# # Pick a user ID
# picked_userid = 1

# # Pick a movie
# picked_movie = 'American Pie (1999)'

# # Movies that the target user has watched
# picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')\
#                           .sort_values(ascending=False))\
#                           .reset_index()\
#                           .rename(columns={1:'rating'})

# picked_userid_watched.head()

# # Similarity score of the movie American Pie with all the other movies
# picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(columns={'American Pie (1999)':'similarity_score'})

# # Rank the similarities between the movies user 1 rated and American Pie.
# n = 5
# picked_userid_watched_similarity = pd.merge(left=picked_userid_watched, 
#                                             right=picked_movie_similarity_score, 
#                                             on='title', 
#                                             how='inner')\
#                                      .sort_values('similarity_score', ascending=False)[:5]

# # Take a look at the User 1 watched movies with highest similarity
# #picked_userid_watched_similarity

# # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
# predicted_rating = round(np.average(picked_userid_watched_similarity['rating'], 
#                                     weights=picked_userid_watched_similarity['similarity_score']), 6)

# print(f'The predicted rating for {picked_movie} by user {picked_userid} is {predicted_rating}' )


# ###### Step 8: Movie Recommendation

# Item-based recommendation function
def item_based_rec(picked_userid, number_of_similar_items=5, number_of_recommendations =3):
  import operator
  # Movies that the target user has not watched
  picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
  picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[picked_userid]==True]['title'].values.tolist()

  # Movies that the target user has watched
  picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')\
                            .sort_values(ascending=False))\
                            .reset_index()\
                            .rename(columns={picked_userid:'rating'})
  
  # Dictionary to save the unwatched movie and predicted rating pair
  rating_prediction ={}  

  # Loop through unwatched movies          
  for picked_movie in picked_userid_unwatched: 
    # Calculate the similarity score of the picked movie iwth other movies
    picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(columns={picked_movie:'similarity_score'})
    # Rank the similarities between the picked user watched movie and the picked unwatched movie.
    picked_userid_watched_similarity = pd.merge(left=picked_userid_watched, 
                                                right=picked_movie_similarity_score, 
                                                on='title', 
                                                how='inner')\
                                        .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
    # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
    predicted_rating = round(np.average(picked_userid_watched_similarity['rating'], 
                                        weights=picked_userid_watched_similarity['similarity_score']), 6)
    # Save the predicted rating in the dictionary
    rating_prediction[picked_movie] = predicted_rating
    # Return the top recommended movies
  return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]

# Get recommendations
recommended_movie = item_based_rec(picked_userid=3, number_of_similar_items=5, number_of_recommendations=4)
print(recommended_movie)
#print(matrix_norm[1].isna())
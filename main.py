import pandas as pd
import numpy as np
import scipy.stats

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

# print(ratings.head())
# print(ratings.info())

# Number of users
# print('The ratings dataset has', ratings['userId'].nunique(), 'unique users')

# # Number of movies
# print('The ratings dataset has', ratings['movieId'].nunique(), 'unique movies')

# # Number of ratings
# print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings')

# # List of unique ratings
# print('The unique ratings are', sorted(ratings['rating'].unique()))

movies = pd.read_csv('data/ml-latest-small/movies.csv')

#print(movies.head())

df = pd.merge(ratings, movies, on='movieId', how='inner')
#print(df.head())

# Aggregate by movie
agg_ratings = df.groupby('title').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()

# Keep the movies with over 100 ratings
agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>100]
#print(agg_ratings_GT100.info())

# Check popular movies
#print(agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head())

# Visulization
#sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
#plt.show()

df_GT100 = pd.merge(df, agg_ratings_GT100[['title']], on='title', how='inner')
#print(df_GT100.info())

# # Number of users
# print('The ratings dataset has', df_GT100['userId'].nunique(), 'unique users')

# # Number of movies
# print('The ratings dataset has', df_GT100['movieId'].nunique(), 'unique movies')

# # Number of ratings
# print('The ratings dataset has', df_GT100['rating'].nunique(), 'unique ratings')

# # List of unique ratings
# print('The unique ratings are', sorted(df_GT100['rating'].unique()))

matrix = df.pivot_table(index='userId', columns='title', values='rating')
#print(matrix.head())

# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
#print(matrix_norm.head())

user_similarity = matrix_norm.T.corr()
#print(user_similarity.head())

user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
#print(user_similarity_cosine)

# Pick a user ID
picked_userid = 8

# Remove picked user ID from the candidate list
user_similarity.drop(index=picked_userid, inplace=True)

# Number of similar users
n = 105

# User similarity threashold
user_similarity_threshold = 0.3

# Get top n similar users
similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid].sort_values(ascending=False)[:n]

# Print out top n similar users
#print(f'The similar users for user {picked_userid} are', similar_users)

picked_userid_watched = matrix_norm[matrix_norm.index == picked_userid].dropna(axis=1, how='all')
#print(picked_userid_watched)

similar_user_movies = matrix_norm[matrix_norm.index.isin(similar_users.index)].dropna(axis=1, how='all')
#print(similar_user_movies)

# Remove the watched movie from the movie list
similar_user_movies.drop(picked_userid_watched.columns,axis=1, inplace=True, errors='ignore')

# Take a look at the data
#print(similar_user_movies)

# A dictionary to store item scores
item_score = {}

# Loop through items
for i in similar_user_movies.columns:
  # Get the ratings for movie i
  movie_rating = similar_user_movies[i]
  # Create a variable to store the score
  total = 0
  # Create a variable to store the number of scores
  count = 0
  # Loop through similar users
  for u in similar_users.index:
    # If the movie has rating
    if pd.isna(movie_rating[u]) == False:
      # Score is the sum of user similarity score multiply by the movie rating
      score = similar_users[u] * movie_rating[u]
      # Add the score to the total score for the movie so far
      total += score
      # Add 1 to the count
      count +=1
  # Get the average score for the item
  item_score[i] = total / count

# Convert dictionary to pandas dataframe
item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])
    
# Sort the movies by score
ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)

# Select top m movies
m = 10
#print(ranked_item_score.head(m))

# Average rating for the picked user
avg_rating = matrix[matrix.index == picked_userid].T.mean()[picked_userid]

# Print the average movie rating for user 1
#print(f'The average movie rating for user {picked_userid} is {avg_rating:.2f}')

# Calcuate the predicted rating
ranked_item_score['predicted_rating'] = ranked_item_score['movie_score'] + avg_rating

# Take a look at the data
print(ranked_item_score.head(m))
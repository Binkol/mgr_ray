import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import ray
import operator


class ItemBasedCF():
    def __init__(self, ratings, movies, number_of_similar_items, number_of_recommendations):
        self.ratings = ratings
        self.movies = movies
        self.df = pd.merge(ratings, movies, on='movieId', how='inner')
        self.num_of_sim_items = number_of_similar_items
        self.num_of_recs = number_of_recommendations

    def prepare_data(self):
        agg_ratings = self.df.groupby('title').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
        agg_ratings_GT25 = agg_ratings[agg_ratings['number_of_ratings']>25]
        df_GT25 = pd.merge(self.df, agg_ratings_GT25[['title']], on='title', how='inner')
        matrix = df_GT25.pivot_table(index='title', columns='userId', values='rating')
        self.matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
        self.item_similarity = self.matrix_norm.T.corr()


    def generateRecomendations(self, user_id):
        #prepare data
        self.prepare_data()

        # Movies that the target user has not watched
        picked_userid_unwatched = pd.DataFrame(self.matrix_norm[user_id].isna()).reset_index()
        picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[user_id]==True]['title'].values.tolist()

        # Movies that the target user has watched
        picked_userid_watched = pd.DataFrame(self.matrix_norm[user_id].dropna(axis=0, how='all')\
                                    .sort_values(ascending=False))\
                                    .reset_index()\
                                    .rename(columns={user_id:'rating'})
        
        # Dictionary to save the unwatched movie and predicted rating pair
        rating_prediction ={}  

        # Loop through unwatched movies          
        for picked_movie in picked_userid_unwatched: 
            # Calculate the similarity score of the picked movie iwth other movies
            picked_movie_similarity_score = self.item_similarity[[picked_movie]].reset_index().rename(columns={picked_movie:'similarity_score'})
            # Rank the similarities between the picked user watched movie and the picked unwatched movie.
            picked_userid_watched_similarity = pd.merge(left=picked_userid_watched, 
                                                        right=picked_movie_similarity_score, 
                                                        on='title', 
                                                        how='inner')\
                                                .sort_values('similarity_score', ascending=False)[:self.num_of_sim_items]
            # Calculate the predicted rating using weighted average of similarity scores and the ratings from user
            predicted_rating = round(np.average(picked_userid_watched_similarity['rating'], 
                                                weights=picked_userid_watched_similarity['similarity_score']), 6)
            # Save the predicted rating in the dictionary
            rating_prediction[picked_movie] = predicted_rating
            # Return the top recommended movies
        return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:self.num_of_recs]


if __name__ == "__main__":
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    movies = pd.read_csv('data/ml-latest-small/movies.csv')
    ibcf = ItemBasedCF(ratings,
                       movies,
                       number_of_similar_items=5,
                       number_of_recommendations=3)
    
    rec = ibcf.generateRecomendations(1)
    print(rec)
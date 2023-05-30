import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import ray

@ray.remote
class UserBasedCF():
    def __init__(self, ratings, movies, numberOfSimilarUsers, similarityThreshold):
        self.ratings = ratings
        self.movies = movies
        self.df = pd.merge(self.ratings, self.movies, on='movieId', how='inner')
        self.userItemMatrix = None
        self.normUserItemMatrix = None
        self.numberOfSimilarUsers = numberOfSimilarUsers
        self.similarityThreshold = similarityThreshold
        self.createMatrixes()

    def createUserItemMatrix(self):
        return self.df.pivot_table(index='userId', columns='title', values='rating')

    def normalizeUserItemMatix(self):
        return self.userItemMatrix.subtract(self.userItemMatrix.mean(axis=1), axis='rows')

    def createMatrixes(self):
        self.userItemMatrix = self.createUserItemMatrix()
        self.normUserItemMatrix = self.normalizeUserItemMatix()

    #algorithm values ['corr', 'cos']
    def createSimilarityMatrix(self, algorithm): 
        if algorithm == "corr":
            return self.normUserItemMatrix.T.corr()
        elif algorithm == "cos":
            return cosine_similarity(self.normUserItemMatrix.fillna(0))


    def getSimilarUsers(self, userID, simMatrix):
        return simMatrix[simMatrix[userID]>self.similarityThreshold][userID].sort_values(ascending=False)[:self.numberOfSimilarUsers]
        

    def removeAlreadyWatchedMovies(self, userID, simUsers):
        userAlreadyWatched = self.normUserItemMatrix[self.normUserItemMatrix.index == userID].dropna(axis=1, how='all')
        similarUserWatched = self.normUserItemMatrix[self.normUserItemMatrix.index.isin(simUsers.index)].dropna(axis=1, how='all')
        similarUserWatched.drop(userAlreadyWatched.columns, axis=1, inplace=True, errors='ignore')
        return similarUserWatched


    def getRecomendations(self, simUsers, simUserMovies):
        item_score = {}

        for i in simUserMovies.columns:
            movie_rating = simUserMovies[i]
            total = 0
            count = 0
            for u in simUsers.index:
                if pd.isna(movie_rating[u]) == False:
                    score = simUsers[u] * movie_rating[u]
                    total += score
                    count +=1
            item_score[i] = total / count

        item_score = pd.DataFrame(item_score.items(), columns=['movie', 'movie_score'])
        ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)
        return ranked_item_score


    def generateRecomendations(self, userID):
        userSimilarityMatrix = self.createSimilarityMatrix("corr") #tego chyba nie trzeba dla kazdego robic?
        similarUsers = self.getSimilarUsers(userID, userSimilarityMatrix)
        similarUserMovies = self.removeAlreadyWatchedMovies(userID, similarUsers)
        recomendations = self.getRecomendations(similarUsers, similarUserMovies)
        return recomendations




if __name__ == "__main__":
    UBCF = UserBasedCF('data/ml-latest-small/ratings.csv',
                       'data/ml-latest-small/movies.csv',
                       numberOfSimilarUsers=10,
                       similarityThreshold=0.3)
    
    #reco = UBCF.generateRecomendations(2)
    #print(reco.head(10))
    for x in range(1,10):
        print("==========STARTING {} =========".format(x))
        reco = UBCF.generateRecomendations(x)
        print(reco.head(5))
        #print("{}/100 done".format(x))
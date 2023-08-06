import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class CBR:
    def __init__(self, movies):
        self.movies = movies
        self.df = None
        self.tf_idf_matrix = None
        self.cosine_similarity_matrix = None
        self.prepareData()

    def getDf(self):
        df_data = self.movies[self.movies['vote_count'].notna()]
        min_votes = np.percentile(df_data['vote_count'].values, 85)
        df = df_data.copy(deep=True).loc[df_data['vote_count'] > min_votes]
        # removing rows with missing overview
        df = df[df['overview'].notna()]
        df.reset_index(inplace=True)
        return df

    def process_text(self, text):
        # replace multiple spaces with one
        text = ' '.join(text.split())
        # lowercase
        text = text.lower()
        return text

    def getTfIdfMatrix(self):
        self.df['overview'] = self.df.apply(lambda x: self.process_text(x.overview),axis=1)
        tf_idf = TfidfVectorizer(stop_words='english')
        tf_idf_matrix = tf_idf.fit_transform(self.df['overview'])
        return tf_idf_matrix

    def index_from_title(self, df, title):
        #print(df[df['original_title']==title].index.values[0], "BBBBBBB")
        return df[df['original_title']==title].index.values[0]

    # function that returns the title of the movie from its index
    def title_from_index(self, df, index):
        return df[df.index==index].original_title.values[0]

    def prepareData(self):
        self.df = self.getDf()
        self.tf_idf_matrix = self.getTfIdfMatrix()
        self.cosine_similarity_matrix = cosine_similarity(self.tf_idf_matrix, self.tf_idf_matrix)

    # generating recommendations for given title
    def recommendations(self, original_title, number_of_recommendations):
        index = self.index_from_title(movies,original_title)
        similarity_scores = list(enumerate(self.cosine_similarity_matrix[index]))
        similarity_scores_sorted = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommendations_indices = [t[0] for t in similarity_scores_sorted[1:(number_of_recommendations+1)]]
        return self.df['original_title'].iloc[recommendations_indices]
    
if __name__ == "__main__":
    movies = pd.read_csv('data/kaggle/movies_metadata.csv', low_memory=False)
    
    half_rows = len(movies) // 2
    half_movies = movies.iloc[:half_rows].copy()

    quater_rows = len(movies) // 4
    quater_movies = movies.iloc[:quater_rows].copy()

    titles = movies.original_title.values.tolist()
    print(quater_movies.info())
    
    cbr = CBR(half_movies)
    for title in titles[0:1000]:
        print("AAAAAA",title)
        rec = cbr.recommendations(title, 5)
        print(rec)
    
    # rec = cbr.recommendations("Waiting to Exhale", 5)
    # #print(rec)
    # print(cbr.index_from_title(movies, "Waiting to Exhale"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ray

@ray.remote(num_cpus=1)
class CBR:
    def __init__(self, metadata):
        self.metadata = metadata
        self.prepareMetadata()
        self.tfidf_matrix = self.get_tfidf()
        self.cosine_sim = self.get_cosine_sim()
        self.indices = pd.Series(self.metadata.index, index=self.metadata['title']).drop_duplicates()

    def prepareMetadata(self):
        self.metadata['overview'] = self.metadata['overview'].fillna('')

    def get_tfidf(self):
        tfidf = TfidfVectorizer(stop_words='english')
        return tfidf.fit_transform(self.metadata['overview'])

    def get_cosine_sim(self):
        return cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, title, rec_num):
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        try:
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        except:
            return "Error - sim score for movie failed"
        sim_scores = sim_scores[1:rec_num]
        movie_indices = [i[0] for i in sim_scores]
        return self.metadata['title'].iloc[movie_indices]


if __name__ == "__main__":
    metadata = pd.read_csv('/root/data/kaggle/movies_metadata.csv', low_memory=False)
    quater_rows = len(metadata) // 2 # /4 = 11k
    metadata = metadata.iloc[:quater_rows].copy()

    titles = metadata.title.values.tolist()

    cbr = CBR(metadata)
    for title in titles:
        rec = cbr.get_recommendations(title, 5)
        print(rec)
import ray
import time
from ubcf import UserBasedCF
from ibcf import ItemBasedCF
from cbr import CBR
import pandas as pd
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if __name__ == "__main__":
    ray.init(_node_ip_address='139.144.77.13')

    # ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    # movies = pd.read_csv('data/ml-latest-small/movies.csv')

    ratings = pd.read_csv('~/data/kaggle/ratings_small.csv')
    kaggle_movies = pd.read_csv('~/data/kaggle/movies_metadata.csv', low_memory=False)
    movies = kaggle_movies[['id','title']]
    movies = movies.rename(columns = {'id':'movieId'})
    movies = movies[(movies.movieId != "1997-08-20") & 
                   (movies.movieId != "2012-09-29") & 
                   (movies.movieId != "2014-01-01")]
    movies = movies.astype({'movieId': 'int64'})

    movies10k = movies[movies['movieId'] <= 10000]
    movies20k = movies[movies['movieId'] <= 20000]
    movies30k = movies[movies['movieId'] <= 30000]

    ratings150 = ratings[ratings['userId'] <= 150]
    ratings300 = ratings[ratings['userId'] <= 300]
    ratings600 = ratings[ratings['userId'] <= 600]

    half_rows = len(movies) // 2
    quater_rows = len(movies) // 4
    
    half_movies = movies.iloc[:half_rows].copy()
    quater_movies = movies.iloc[:quater_rows].copy()

    titles = movies.title.values.tolist()
    
    start_time = time.time()

    cluster_size = 2
    actors_size = cluster_size*4
    pg_list = [{"CPU": 4} for _ in range(cluster_size)]
    pg = placement_group(pg_list, strategy="SPREAD")
    
    try:
        ray.get(pg.ready(), timeout=10)
    except Exception as e:
        print("Cannot create a placement group because ")
        print(e)

    # actors = [UserBasedCF.remote(ratings, # <--
    #                             movies,
    #                             numberOfSimilarUsers=10,
    #                             similarityThreshold=0.3) for _ in range(actors_size)]
    
    # actors = [ItemBasedCF.remote(ratings, 
    #                            movies, # <--
    #                            number_of_similar_items=5,
    #                            number_of_recommendations=3) for _ in range(actors_size)]

    actors = [CBR.remote(kaggle_movies) for _ in range(actors_size)]


    run_cbr = True

    futures = []
    if run_cbr:
        for i, title in enumerate(titles[0:300]):
            actor_id = (i+1)%actors_size
            ref = actors[actor_id].recommendations.remote(title, 5)
            futures.append(ref)
    else:
        for id in range(1,300):
            actor_id = id%actors_size
            ref = actors[actor_id].generateRecomendations.remote(id)
            futures.append(ref)

    #output = ray.get(futures)
    
    counter = 1
    for x in ray.get(futures):
        print("=== {} recomendation ===".format(counter))      
        #print(x.head(5))
        counter += 1

    #print(ray.get(output[88]).head(5))
    print(time.time() - start_time)

    

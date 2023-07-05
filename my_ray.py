import ray
import time
from cbcf import UserBasedCF
from ibcf import ItemBasedCF
import pandas as pd
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if __name__ == "__main__":
    ray.init(_node_ip_address='192.168.1.70')

    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    movies = pd.read_csv('data/ml-latest-small/movies.csv')

    start_time = time.time()

    cluster_size = 4
    actors_size = cluster_size*4
    pg_list = [{"CPU": 4} for _ in range(cluster_size)]
    pg = placement_group(pg_list, strategy="SPREAD")
    
    try:
        ray.get(pg.ready(), timeout=10)

        # actors = [UserBasedCF.remote(ratings,
        #                             movies,
        #                             numberOfSimilarUsers=10,
        #                             similarityThreshold=0.3) for _ in range(actors_size)]
        
        actors = [ItemBasedCF.remote(ratings,
                                    movies,
                                    number_of_similar_items=5,
                                    number_of_recommendations=3) for _ in range(actors_size)]


        futures = []
        for id in range(1,300):
            actor_id = id%actors_size
            ref = actors[actor_id].generateRecomendations.remote(id)
            futures.append(ref)
            #time.sleep(0.5)

        #output = ray.get(futures)
        
        counter = 1
        for x in ray.get(futures):
            print("=== {} recomendation ===".format(counter))      
            #print(x.head(5))
            counter += 1
    
        #print(ray.get(output[88]).head(5))
        print(time.time() - start_time)

    except Exception as e:
        print("Cannot create a placement group because ")
        print(e)

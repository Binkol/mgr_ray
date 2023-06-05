import ray
import time
from cbcf import UserBasedCF
import pandas as pd
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if __name__ == "__main__":
    ray.init(_node_ip_address='192.168.1.70')
    cluster_size = 4

    ratings = []
    movies = []
    for data_id in range(1, cluster_size+1):
        ratings.append(pd.read_csv('data{}/ml-latest-small/ratings.csv'.format(data_id)))
        movies.append(pd.read_csv('data{}/ml-latest-small/movies.csv'.format(data_id)))

    pg_list = [{"CPU": 4} for _ in range(cluster_size)]
    pg = placement_group(pg_list, strategy="SPREAD")
    
    start_time = time.time()
    try:
        ray.get(pg.ready(), timeout=10)

        actors = [UserBasedCF.remote(ratings[id],
                                    movies[id],
                                    numberOfSimilarUsers=10,
                                    similarityThreshold=0.3) for id in range(cluster_size)]
        

        futures = []
        for id in range(1,100):
            cluster_id = id%cluster_size
            ref = actors[cluster_id].generateRecomendations.remote(id)
            futures.append(ref)
            time.sleep(0.5)

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

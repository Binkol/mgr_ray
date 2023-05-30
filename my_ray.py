import ray
import time
from cbcf import UserBasedCF
import pandas as pd

@ray.remote
def calc(ubcf, userid):
    reco = ubcf.generateRecomendations.remote(userid)
    return reco

if __name__ == "__main__":
    ray.init(_node_ip_address='192.168.1.163')

    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    movies = pd.read_csv('data/ml-latest-small/movies.csv')

    start_time = time.time()

    cluster_size = 2
    actors = [UserBasedCF.remote(ratings,
                                movies,
                                numberOfSimilarUsers=10,
                                similarityThreshold=0.3) for _ in range(cluster_size)]
    

    #futures = [calc.remote(UBCF, i) for i in range(1,10)]
    futures = []
    for id in range(1,100):
        cluster_id = id%cluster_size
        ref = calc.remote(actors[cluster_id], id)
        futures.append(ref)
        #time.sleep(0.5)

    output = ray.get(futures)
    
    counter = 1
    for x in output:
        print("=== {} recomendation ===".format(counter))      
        print(ray.get(x).head(5))
        counter += 1
  
    #print(ray.get(output[88]).head(5))
    print(time.time() - start_time)

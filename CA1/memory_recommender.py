from similarity import Similarity


class MemoryRecommender:
    """Implement the required functions for Q2"""
    def __init__(self, sim_func, weight_type="item-based"):
        #TODO: implement any necessary initalization function here
        # sim_func is one of the similarity functions implemented in similarity.py 
        # weight_type is one of ["item-based", "user-based"]
        #You can add more input parameters as needed.
        pass
    
    def rating_predict(self, userID, itemID):
        #TODO: implement the rating prediction function for a given user-item pair 
        pass
    
    def topk(self, userID, k=5):
        #TODO: implement top-k recommendations for a given user
        pass
        

if __name__ == '__main__':
    
    sim = Similarity()
    # print the solution to Q3a here
    recommender = MemoryRecommender(sim.cosine_similarity, weight_type="item-based")
    recommender.topk(userID=381)
    
    recommender = MemoryRecommender(sim.pearson_similarity, weight_type="item-based")
    recommender.topk(userID=381)
    
    # print the solution to Q3b here
    recommender = MemoryRecommender(sim.cosine_similarity, weight_type="user-based")
    recommender.topk(userID=381)
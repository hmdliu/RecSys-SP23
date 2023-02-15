class Similarity:
    """Implement the required functions for Q2"""
    def __init__(self):
        #TODO: implement any necessary initalization function here
        #You can add more input parameters as needed.
        pass
    
    def jaccard_similarity(self, item1, item2):
        #TODO: implement the required functions and print the solution to Question 2a here
        pass
        
    def cosine_similarity(self, item1, item2):
        #TODO: implement the required functions and print the solution to Question 2b here
        pass
    
    def pearson_similarity(self, item1, item2):
        #TODO: implement the required functions and print the solution to Question 2c here
        pass
        

if __name__ == '__main__':
    
    sim = Similarity()
    
    # print the solution to Q2a here
    sim.jaccard_similarity(item1=1, item2=2)
    sim.jaccard_similarity(item1=1, item2=3114)
    
    # print the solution to Q2b here
    sim.cosine_similarity(item1=1, item2=2)
    sim.cosine_similarity(item1=1, item2=3114)
    
    # print the solution to Q2c here
    sim.pearson_similarity(item1=1, item2=2)
    sim.pearson_similarity(item1=1, item2=3114)
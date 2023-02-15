class RecDataset:
    """Implement the required functions for Q1"""
    def __init__(self):
        #TODO: implement any necessary initalization function such as data loading here
        #You can add more input parameters as needed.
        pass
    
    def describe(self):
        #TODO: implement the required functions and print the solution to Question 1a here
        pass
        
    def query_user(self, userID):
        #TODO: implement the required functions and print the solution to Question 1b here
        pass
    
    def dist_by_age_groups(self):
        #TODO: implement the required functions and print the solution to Question 1c here
        #You could import `users.dat` here or in __init__(). 
        #This function is expected to return two lists - you shall use these lists to 
        #draw the bar plots and attach them in your answer sheet.
        pass
        

if __name__ == '__main__':
    
    dataset = RecDataset()
    
    # print the solution to Q1a here
    dataset.describe() 
    
    # print the solution to Q1b here
    dataset.query_user(userID=100)
    dataset.query_user(userID=381) 
    
    # print the solution to Q1c here
    dataset.dist_by_age_groups()
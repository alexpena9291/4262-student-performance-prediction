#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
class K_Fold():
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        
    def k_fold_test(self):
        #Split up data into folds
        listX = np.array_split(self.X, self.k, axis=0)
        listY = np.array_split(self.y, self.k)

        #Set up Model
        model = LinearRegression()
        
        #get average accuracy
        total = 0
        for i in range(self.k):
            #Partition matricies to use
            training_X = np.vstack(np.delete(listX, i, 0))
            training_Y = np.vstack(np.delete(listY, i, 0))
            
            model.fit(training_X, np.ravel(training_Y))
            y_pred = model.predict(listX[i])
            error = mean_squared_error(y_pred, listY[i])
            total += error

        return total/self.k


# In[ ]:





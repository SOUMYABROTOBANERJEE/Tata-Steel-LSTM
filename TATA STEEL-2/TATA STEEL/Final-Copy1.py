#!/usr/bin/env python
# coding: utf-8

def Trainer(path):
    import pandas as pd
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import numpy as np
    
    # In[2]:
    
    import warnings
    warnings.filterwarnings('ignore')
    
    ################################################# MULTI CLASS #####################################################
    
    
    df1=pd.read_excel(path,'Sheet1')
    df2=df1[['Net Due Dt','Clearing','Status']]
    df2.dropna(inplace=True)
    df2['Range of Delay']=df2['Clearing']-df2['Net Due Dt']
    
    
    # In[8]:
    
    
    df2['Range of Delay']=df2['Range of Delay'].dt.days
    #df['Difference'] = pd.to_numeric(df['Difference']) 
    df2['Range of Delay']=pd.to_numeric(df2['Range of Delay'])
    #df2['Range of Delay']
    
    
    # In[9]:
    
    
    df2 = df2.reset_index()
    df2.drop(columns='index',inplace=True)
    
    df2['Status'] = np.where(df2['Range of Delay'] > 180, 8, 
             (np.where(df2['Range of Delay'] > 90, 7,(np.where(df2['Range of Delay'] > 60, 6, (np.where(df2['Range of Delay'] > 30, 5, (np.where(df2['Range of Delay'] > 15, 4, (np.where(df2['Range of Delay'] > 7, 3, (np.where(df2['Range of Delay'] > 3, 2, (np.where(df2['Range of Delay'] > 0, 1 ,0)))))))))))))))
    
    
    # In[11]:
    
    
    df2['Status'].value_counts()
    
    
    # In[12]:
    
    
    df1['Status']=df2['Status']
    
    
    # In[13]:
    
    
    df1.head(5)
    
    
    # In[14]:
    
    
    df1.dropna(inplace=True)
    df1['Status']=df1['Status'].astype('category').cat.codes
    df1['    BusA']=df1['    BusA'].astype('category').cat.codes
    df1['CCAr']=df1['CCAr'].astype('category').cat.codes
    #df['Account']=df['Account'].astype('category').cat.codes
    #df1.drop(columns='Reference',inplace=True)
    df1.drop(columns='Customer Name',inplace=True)
    df1.drop(columns=['DocumentNo','Month','Year','Clrng doc.'],inplace=True)
    df1['Zone']=df1['Zone'].astype('category').cat.codes
    df1['Bran']=df1['Bran'].astype('category').cat.codes
    df1['PayT'] = df1['PayT'].astype('category').cat.codes
    df1.drop(columns='Doc/Chq dt',inplace=True)
    df1.head(5)
        
    
    
    # In[15]:
    
    
    df1['Pstng Date']=df1['Pstng Date'].dt.strftime("%Y%m%d").astype(str)
    df1['Net Due Dt']=df1['Net Due Dt'].dt.strftime("%Y%m%d").astype(str)
    df1['Clearing']=df1['Clearing'].dt.strftime("%Y%m%d").astype(str)
    
    
    # In[16]:
    
    
    df1.drop(columns=['Net Due Dt','Clearing'],inplace=True)
    
    
    # In[17]:
    
    
    df1 = df1.reset_index()
    df1.drop(columns='index',inplace=True)
    
    
    # In[18]:
    
    
    df1.drop(columns='Arr (Clearing - Net Due Date)',inplace=True)
    
    
    # In[19]:
    
    
    y=np.array(df1['Status'])
    df1.drop(columns='Status',inplace=True)
    
    
    # In[20]:
    
    
    ##############  normalizing data ##########################
    
    
    # In[21]:
    
    
    from sklearn import preprocessing
    y1=df1.columns
    x = df1.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1 = pd.DataFrame(x_scaled,columns=df1.columns)
    df1.head(5)
    
    
    # In[22]:
    
    
    
    X=df1
    
    
    # In[23]:
    
    
    k,count=np.unique(y,return_counts=True)
    count
    
    
    # In[24]:
    
    
    ############# SPLITTING DATA ###################
    
    
    # In[25]:
    
    
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3)
    
    
    ################### RANDOM FOREST ########################
    
    
    # In[143]:
    
    
    from sklearn.ensemble import RandomForestClassifier
    model= RandomForestClassifier(n_estimators = 100, criterion='entropy',random_state = 0)
    y1=model.fit(xtrain, ytrain)
    y2=model.predict(xtest)
    
    
    # In[144]:
    
    
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score
    print("Accuracy: for Random Forest",metrics.accuracy_score(ytest, y2)*100)
    #print('ROC SCORE:',roc_auc_score(ytest, y2))
    print(pd.Series(model.feature_importances_,index=df1.columns).sort_values(ascending=False))
    
    
    # In[146]:
    
    
    unique, counts = np.unique(ytest, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(y2, return_counts=True)
    print(dict(zip(unique, counts)))
    print()
    print('Confusion Matrix for Random Forest')
    print(metrics.confusion_matrix(y2, ytest))
    print('------------------------------------------------------------------------------------------------')
    
    
    # In[167]:
    
    
    ################### Decision tress  #####################
    
    
    # In[168]:
    
    
    from sklearn import tree
    decision_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=100)
    ytree =decision_tree.fit(xtrain, ytrain)
    ytreepred=decision_tree.predict(xtest)
    print("Accuracy for Decision Trees:",metrics.accuracy_score(ytreepred,ytest)*100)
    
    
    # In[169]:
    
    
    unique, counts = np.unique(ytest, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(ytreepred, return_counts=True)
    print(dict(zip(unique, counts)))
    print()
    print('Confusion Matrix for Decision Trees')
    print(metrics.confusion_matrix(ytreepred, ytest))
    
    
    
    # In[177]:
    
    
    print(pd.Series(decision_tree.feature_importances_,index=df1.columns).sort_values(ascending=False))
    print('------------------------------------------------------------------------------------------------')
    
    # In[39]:
    
    
    ############################################  KNN ##################################
    
    
    # In[120]:
    
    
    from sklearn import neighbors
    
    
    # In[121]:
    
    
    limit=10
    
    
    # In[124]:
    
    
    for i in range(limit):
        mo = neighbors.KNeighborsClassifier(n_neighbors =i+2)
        mo.fit(xtrain, ytrain)
        predict = mo.predict(xtest)
        print("Accuracy: for KNN ",metrics.accuracy_score(predict, ytest)*100,'for k= ',(i+2))
    
    
    # In[ ]:
    
    
    metrics.accuracy_score(predict, ytest)*100
    

    arr,count=np.unique(predict,return_counts=True)
    print(arr,count)
    

    
    print()
    print('Confusion Matrix for KNN')
    print(metrics.confusion_matrix(predict, ytest))
    
    
    # In[54]:
    
    
    unique, counts = np.unique(ytest, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(predict, return_counts=True)
    print(dict(zip(unique, counts)))
    print('---------------------------------------------------------------------------------------------')
    

def predictor(df):
    dt=decision_tree.predict(df)
    rft=model.predict(df)
    kn=mo.predict(df)
    print('Prediction by Decision Tree',dt)
    print('Prediction by Random Forest',rft)
    print('Prediction by KNN',kn)
    return [dt,rft,kn]


# In[193]:



    arr=predictor(xtest[:50])



    print('Accuracy of Decision Trees',metrics.accuracy_score(ytest[:50],arr[0]))
    print('Confusion Matrix: \n',metrics.confusion_matrix(ytest[:50],arr[0]))
    
    
    # In[197]:
    
    
    print('Accuracy of Random Forest',metrics.accuracy_score(ytest[:50],arr[1]))
    print('Confusion Matrix: \n',metrics.confusion_matrix(ytest[:50],arr[1]))
    
    
    # In[199]:
    
    
    print('Accuracy of KNN',metrics.accuracy_score(ytest[:50],arr[2]))
    print('Confusion Matrix: \n',metrics.confusion_matrix(ytest[:50],arr[2]))
    
    
    for i in range(len(arr)):
        for j in range(arr[i]):
            if(arr[i][j]==0):
                print('Payment is predicted to be received on time')
            if (arr[i][j]==1):
                print('Payment is predicted to be received 0-3 days')
            if arr[i][j]==2:
                print('Payment is predicted to be received 3-7 days')
            if arr[i][j]==3:
                print('Payment is predicted to be received 8-15 days')
            if arr[i][j]==4:
                print('Payment is predicted to be received 16-30 days')               
            if arr[i][j]==5:
                print('Payment is predicted to be received 30-60 days')
            if arr[i][j]==6:
                print('Payment is predicted to be received 60-90 days')
            if arr[i][j]==7:
                print('Payment is predicted to be received 90-180 days')
            if arr[i][j]==8:
                print('Payment is predicted to be received >180 days')
        





# In[ ]:





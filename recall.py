def Model(cols):

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import confusion_matrix

    
    if (all(col == 0 for col in cols)):
      return 0
    
    else:
          
      cols_num = []
      for i in range (len(cols)):
        if cols[i] == 1:
            cols_num.append(i)
    
      cols_num.append(11)

      df = pd.read_csv('Encoded_dataset.csv')
      df.drop(['Unnamed: 0'],axis=1,inplace=True)

      df = df.iloc[:,cols_num]
      
      x_train,x_test,y_train,y_test = train_test_split(df.drop(columns = ['HeartDisease'] , axis = 1) , df['HeartDisease'] , test_size = 0.2 , random_state=390)
      clf = xgb.XGBClassifier()
      clf.fit(x_train, y_train)
      y_pred = clf.predict(x_test)
      cm = confusion_matrix(y_test, y_pred)
      cm = np.array(cm)

      # Calculate recall
      if (cm[0][0] == 0 and cm[1][0] == 0):
        recall = 0
      else: 
        recall = cm[0][0] / (cm[0][0] + cm[1][0])
        
      return recall


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')
df=pd.read_csv('diabetes.csv')
col_to_drop=[]
for col in df.columns:
    if df[col].nunique()==df.shape[0] or df[col].nunique()==1:
        col_to_drop.append(df[col])
y=df['Outcome']
x=df.drop(columns='Outcome',inplace=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2,random_state=2)

for col in x_train.columns:
    if x_train[col].dtype=='object':
        x_train[col]=x_train[col].fillna(x_train[col].mode()[0]).astype(object)
        x_test[col]=x_test[col].fillna(x_train[col].mode()[0]).astype(object)
    else:
        x_train[col]=x_train[col].fillna(x_train[col].mean())
        x_test[col]=x_test[col].fillna(x_train[col].mean())


from sklearn.preprocessing import LabelEncoder
import pandas as pd
class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score,accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
le=LabelEncoderExt()
std=MinMaxScaler()
for col in x_train.columns:
    if x_train[col].dtype=='object':
        le.fit(x_train[col])
        x_train[col]=le.transform(x_train[col])
        x_test[col]=le.transform(x_test[col])
    elif x_train[col].dtype=='int64' or x_train[col].dtype=='float64':
        x_train[col]=std.fit_transform(np.array(x_train[col]).reshape(-1,1))
        x_test[col]=std.transform(np.array(x_test[col]).reshape(-1,1))
            
from sklearn.ensemble import GradientBoostingClassifier
gra=GradientBoostingClassifier()
gra.fit(x_train,y_train)
#y_gra_pred=gra.predict(x_test)

'''print(precision_score(y_test,y_gra_pred))
print(recall_score(y_test,y_gra_pred))
print(f1_score(y_test,y_gra_pred))
print(accuracy_score(y_test,y_gra_pred))
'''

filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(gra, open(filename, 'wb'))




















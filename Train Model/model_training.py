import os
import  numpy as np
import cv2
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#Preparing the data
input_dir=os.path.join('../Image_dataset')
categories =['not_parked','parked']

data=[]
labels=[]

for category_idx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path=os.path.join(input_dir,category,file)
        img=cv2.imread(img_path)
        img=resize(img,(15,15))
        data.append(img.flatten()) # Flatten image: From Matrix data to list/array (uni dimentional)
        labels.append(category_idx)

data=np.array(data)
labels= np.array(labels)

#Train / Test split

x_train, x_test , y_train, y_test = train_test_split(data,labels, test_size=0.2,shuffle=True,stratify=labels)
#Startify is used to keep same proportion of distribution of data


# train classifier
classifier = SVC()

parameters = [{'gamma':[0.01,0.001,0.0001], 'C':[1,10,100,1000]}]

grid_search= GridSearchCV(classifier,parameters)

grid_search.fit(x_train,y_train)

#Test Performance
best_estimator = grid_search.best_estimator_

y_predictions=best_estimator.predict(x_test)

score =accuracy_score(y_predictions,y_test)

print('{}% of samples were correctly classified'.format(str(score *100)))

#Save this model

pickle.dump(best_estimator,open('./model.p','wb'))





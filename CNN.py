import os
import cv2
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

DATADIR = "DATA"
CATEGORIES = ["Dirty","Clean"]
IMG_SIZE=64
training_data=[]

for category in CATEGORIES:
    path= os.path.join(DATADIR,category)
    class_num= CATEGORIES.index(category)
    
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        filtered_img_array=cv2.Canny(img_array,100,200)
        resFil_img_array=cv2.resize(filtered_img_array,(IMG_SIZE,IMG_SIZE))
        training_data.append([resFil_img_array, class_num])
        
random.shuffle(training_data)

X=[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)



X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3),activation="relu", input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X, y, batch_size=32, epochs=30, validation_split=0.2)

model.save("CannyEdgeCNN.model")

model =tf.keras.models.load_model("CannyEdgeCNN.model")

def prepare(path,IMG_SIZE=64):
    img_array=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    filtered_img_array=cv2.Canny(img_array,100,200)
    resFil_img_array=cv2.resize(filtered_img_array,(IMG_SIZE,IMG_SIZE))
    return resFil_img_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

predictionClean = model.predict([prepare("DATA\\Test\\1.jpg")])
predictionDirty = model.predict([prepare("DATA\\Test\\2.jpg")])

print("Expected=Clean, Predicted=",CATEGORIES[int(predictionClean[0][0])])
print("Expected=Dirty, Predicted=",CATEGORIES[int(predictionDirty[0][0])])



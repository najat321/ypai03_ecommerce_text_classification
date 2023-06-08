#%%
#1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
# %%
#2. Data loading
PATH = os.path.join(os.getcwd(),"ecommerceDataset.csv")
data = pd.read_csv(PATH)

# %%
#3. Data inspection
print(data.info())
print("-"*20)
print(data.describe())
print("-"*20)
print(data.isna().sum())
print("-"*20)
print(data.duplicated().sum())
#%%
data.dropna(inplace=True)
data.isna().sum()
#%%
data.drop_duplicates(inplace=True)
data.duplicated().sum()
# %%
#4. The review is the feature, the sentiment is the label
feature = data['text'].values
label = data['category'].values
# %%
#5. Convert label into integers using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_processed = label_encoder.fit_transform(label)

# %%
#6. Data preprocessing
#(A) Remove unwanted strings from the data
import re
def remove_unwanted_strings(review):
    for index, data in enumerate(review):
        # Anything within the <> will be removed 
        # ? to tell it dont be greedy so it won't capture everything from the 
        # first < to the last > in the document
        review[index] = re.sub('<.*?>', ' ', data) 
        review[index] = re.sub('[^a-zA-Z]',' ',data).lower().split()
    return review

feature_removed = remove_unwanted_strings(feature)
# %%
#7. Define some hyperparameters
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8
# %%
#8. Perform train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(feature_removed,label_processed,train_size=training_portion,random_state=12345)
# %%
#9. Perform tokenization
from tensorflow import keras

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,split=" ",oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
#%%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
# %%
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
# %%
#10. Perform padding and truncating
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens,maxlen=(max_length))
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens,maxlen=(max_length))
# %%
#11. Model development
#(A) Create the sequential model
model = keras.Sequential()
#(B) Create the input layer, in this case, it can be the embedding layer
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Create the bidirectional LSTM layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)))
#(C) Classification layers
model.add(keras.layers.Dense(embedding_dim,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(y_train)),activation='softmax'))

model.summary()
# %%
#12. Model compilation
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#%%
#13. Create a TensorBoard callback object for the usage of TensorBoard
import datetime
from tensorflow.keras import callbacks
base_log_path = r"tensorboard_logs\ecommerce_text_classification"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#14. Model training
history = model.fit(X_train_padded,y_train,validation_data=(X_test_padded,y_test),epochs=5,batch_size=64,callbacks=[tb])
# %%
#15. Model evaluation
print(history.history.keys())
# %%
#Plot accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train accuracy","Test accuracy"])
plt.show()
# %%
#16. Model deployment
# test_string = ['This house needs some decorating']

#%%
# test_string_removed = remove_unwanted_strings(test_string)
# #%%
# test_string_tokens = tokenizer.texts_to_sequences(test_string_removed)
# #%%
# test_string_padded = keras.preprocessing.sequence.pad_sequences(test_string_tokens,maxlen=(max_length))

# %%
#Make prediction on the test set
y_pred = np.argmax(model.predict(X_test_padded),axis=1)
#%%
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score",f1)
# %%
label_map = ["Household","Books","Clothing & Accessories","Electronics"]
predicted_sentiment = [label_map[i] for i in y_pred]
# %%
#17. Save model and tokenizer
PATH1 = r"C:\Users\USER\Desktop\Deep_learning_sequential\ecommerce_text_classification\saved_models"
print(PATH1)
# %%
#Model save path
model_save_path = os.path.join(PATH1,"ecommerce_text_classification_model.h5")
keras.models.save_model(model,model_save_path)
#%%
#Check if the model can be loaded
model_loaded = keras.models.load_model(model_save_path)
# %%
#tokenizer save path
import pickle

tokenizer_save_path = os.path.join(PATH1,"tokenizer.json")
with open(tokenizer_save_path,'wb') as f:
    pickle.dump(tokenizer,f)
# %%
#Check if the tokenizer object can be loaded
with open(tokenizer_save_path,'rb') as f:
    tokenizer_loaded = pickle.load(f)
# %%
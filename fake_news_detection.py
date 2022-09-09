#!/usr/bin/env python
# coding: utf-8

# In[27]:


# including all libraries and packages used

import transformers
from transformers import BertTokenizer, BertModel
import cv2
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.metrics import categorical_crossentropy
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd


# In[2]:


# Instantiating Bert Tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[3]:


# Instantiating Bert Model

bert_model = BertModel.from_pretrained("bert-base-uncased")


# In[4]:


# Instantiating YOLO 

yolo =  cv2.dnn.readNet('yolov3.weights','yolov3.cfg')


# In[5]:


# Reading calsses lablels from YOLO

classes = []

with open('coco.names', 'r') as f:
      classes = f.read().splitlines()


# In[21]:


# Constants used for Dataset

BATCH_SIZE = 500
TOTAL_TRAINING_SIZE= 20000
TESTING_SET_START_INDEX = 20000
TESTING_SET_END_INDEX = 24999


# In[5]:


# Dumping Bert model in model file

with open('model', 'wb') as files:
    pickle.dump(bert_model, files)


# In[6]:


# Loading Bert model from file

with open('model' , 'rb') as f:
    lr = pickle.load(f)


# In[7]:


# Reading data set

df = pd.read_csv('multimodal_train.tsv', sep='\t')
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)


# In[10]:


# Generating model to be trained

output_model = Sequential([
    Dense(5, input_shape = (23040,), activation = 'relu'),
    Dense(2, activation = 'softmax'),
])


# In[8]:


# Implementing attention function

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


# In[9]:


# Extracting labels from the images from start_index to end_index of the dataset

def yolo_label_extraction(start_index, end_index):
    
    # final_obj_list contains the objects of image
    final_obj_list = []
    final_vec_list = [] 
    
    for i in range(start_index, end_index + 1) :
        if df.loc[i, "hasImage"] == True and df.loc[i, "image_url"] != "" and df.loc[i, "image_url"] != "nan":
            image = cv2.imread("images/"+str(i)+".jpg")
            if image is None:
                final_obj_list.append([])
                i = i + 1
                continue

            blob = cv2.dnn.blobFromImage(image, 1 / 255, (320,320),(0,0,0),swapRB = True, crop = False)
            # Print image
            t = blob[0].reshape(320, 320, 3)
            #plt.imshow(t)
            yolo.setInput(blob)
            output_layers_name = yolo.getUnconnectedOutLayersNames()
            layeroutput = yolo.forward(output_layers_name)
            class_ids = []

            for output in layeroutput:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > 0.7:
                        class_ids.append(class_id)

            list_obj = []     
            for id in class_ids:
                if classes[id] not in list_obj:
                    list_obj.append(classes[id])
            final_obj_list.append(list_obj)
        else:
            final_obj_list.append([])

            
    return final_obj_list


# In[10]:


# Function to convert images and text into vector

def preprocessing_of_test_training(start_index, end_index):
    label = df['2_way_label']
    label2 = label[start_index : end_index + 1]
    
    
    inputs=[]
    outputs=[]
    for i in range(end_index - start_index + 1):
        inputs.append(tokenizer(df['clean_title'][start_index + i], return_tensors="pt"))
        outputs.append(bert_model(**inputs[i]))
        
        
    last_hidden_states=[]
    for i in range(end_index - start_index + 1):
        last_hidden_states.append(outputs[i].last_hidden_state)

    
    final_obj_list = yolo_label_extraction(start_index, end_index)
    print(len(final_obj_list))
    
    inputs1=[]
    outputs1=[]
    for i in range(end_index - start_index + 1):
        st = ""
        for it in final_obj_list[i]:
            st = st + " " + it
        inputs1.append(tokenizer(st, return_tensors="pt"))
        outputs1.append(bert_model(**inputs1[i]))
        
        
    last_hidden_states1=[]
    for i in range(end_index - start_index + 1):
        last_hidden_states1.append(outputs1[i].last_hidden_state)
    
    
    a1 = []
    for i in range(end_index - start_index + 1):
        a1.append(last_hidden_states[i][0])

    a2 = []
    for i in range(end_index - start_index + 1):
        a2.append(last_hidden_states1[i][0])
        
    text1=[]
    for i in range(len(a1)):
        text1.append(a1[i].cpu().detach().numpy())
        
    max_num_of_words=0
    for i in range (len(a1)):
        x = len(a1[i])
        if (x >max_num_of_words):
            max_num_of_words=x
    max_num_of_words
    
    
    final_text=[]
    for i in range (end_index - start_index + 1):
        final_text.append(np.pad(text1[i], ((0,max_num_of_words-len(text1[i])),(0,0)), 'constant'))
        
        
    image1=[]
    for i in range(len(a2)):
        image1.append(a2[i].cpu().detach().numpy())
        
        
    max_num_of_labels_of_image = 0
    for i in range (len(a2)):
        x=len(a2[i])
        if (x >max_num_of_labels_of_image):
            max_num_of_labels_of_image=x
    max_num_of_labels_of_image
    
    final_image=[]
    for i in range (end_index - start_index + 1):
        final_image.append(np.pad(image1[i], ((0,max_num_of_labels_of_image-len(image1[i])),(0,0)), 'constant'))
        
    out, temp_attn = scaled_dot_product_attention(final_image, final_text, final_text, None)
    
    
    
    final_output=[]
    for i in range(len(out)):
        final_output.append(out[i].numpy())
    

    flatten_output = []

    for i in range(end_index - start_index + 1):
        list1 = final_output[i].flatten()
        if(len(list1) < 23040):
            for i in range(23040 - len(list1)):
                list1 = np.append(list1,0)

        else:
            list1 = list1[:23040]
        
        flatten_output.append(list1)
    
    array = np.array(flatten_output)
   
    return (array, label2)

   


# In[11]:


# Function for traning the model

def training_model(input_array, output_label):
    output_model.compile(Adam(lr = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    output_model.fit(input_array, output_label, batch_size = 10, epochs = 25, shuffle = True)
    


# In[23]:


# Function for testing the model

def testing_model(testing_array, output_label):
    predictions = output_model.predict(testing_array,batch_size=10,verbose=10)

    prediction_label=[]
    for i in predictions:
        if(i[0]> i[1]):
            prediction_label.append(0)
        else:
            prediction_label.append(1)
   
    
    acc=accuracy_score(output_label, prediction_label)
    pre=precision_score(output_label, prediction_label)
    rec=recall_score(output_label, prediction_label)
    return (acc,pre,rec)


# In[18]:


# Traning model on the dataset batchwise

for i in range(int(TOTAL_TRAINING_SIZE / BATCH_SIZE)):
    (preprocessed_input, preprocessed_output) = preprocessing_of_test_training(i*BATCH_SIZE,(i+1)*BATCH_SIZE-1)
    
    training_model(preprocessed_input, preprocessed_output)
    


# In[24]:


# Testing model batch wise and calculating accuracy precession and recall

accuracy = 0
precession = 0
recall = 0

TOTAL_TESTING_SIZE=TESTING_SET_END_INDEX-TESTING_SET_START_INDEX 
for i in range(int(TOTAL_TESTING_SIZE / BATCH_SIZE)):
    preprocessed_input, preprocessed_output = preprocessing_of_test_training(TESTING_SET_START_INDEX + i * BATCH_SIZE, TESTING_SET_START_INDEX + (i+1) * BATCH_SIZE-1)

    temp = testing_model(preprocessed_input, preprocessed_output)
    accuracy = accuracy+ temp[0];
    precession = precession +temp[1]
    recall = recall+temp[2]
    
accuracy=(accuracy * BATCH_SIZE/TOTAL_TESTING_SIZE)*100
precession=(precession * BATCH_SIZE/TOTAL_TESTING_SIZE)*100
recall=(recall* BATCH_SIZE /TOTAL_TESTING_SIZE)*100

 


# In[26]:


# Printing the accuracy precession recall and F1 score

print("Accuracy : ",accuracy)
print("Precessoin : ",precession)
print("Recall : ",recall)
print("F1 Score : ", 2 * precession * recall / (precession + recall))


# In[28]:


# Storing trained model into file

with open('final_model', 'wb') as files:
    pickle.dump(output_model, files)


# In[13]:


# Loading already trained model from the file

with open('final_model' , 'rb') as f:
    output_model = pickle.load(f)


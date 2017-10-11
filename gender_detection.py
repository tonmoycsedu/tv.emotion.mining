
# coding: utf-8

# ##Age and Gender Classification Using Convolutional Neural Networks - Demo
# 
# This code is released with the paper:
# 
# Gil Levi and Tal Hassner, "Age and Gender Classification Using Convolutional Neural Networks," IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
# 
# If you find the code useful, please add suitable reference to the paper in your work.

# ## Loading the mean image

# In[1]:

import caffe


# In[2]:

import os
import numpy as np
import matplotlib.pyplot as plt



#caffe_root = './caffe/' 
#import sys
#sys.path.insert(0, caffe_root + 'python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[3]:

mean_filename='./gender_age_detection/mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]


# ## Loading the age network

# In[4]:

# age_net_pretrained='./age_net.caffemodel'
# age_net_model_file='./deploy_age.prototxt'
# age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
#                        mean=mean,
#                        channel_swap=(2,1,0),
#                        raw_scale=255,
#                        image_dims=(256, 256))


# ## Loading the gender network

# In[4]:

gender_net_pretrained='./gender_age_detection/gender_net.caffemodel'
gender_net_model_file='./gender_age_detection/deploy_gender.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


# ## Labels

# In[5]:

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']


# ## Reading and plotting the input image

# In[7]:

#example_image = './images/images-10000000_1463349307034642_1694941330775474176_n.mp4/Img149.jpg'
#example_image = './images/2017 natok/lux-natok/Shesher Golpo _ Full Drama _ Lux Chirochena Shourobher Golpo-DerWLNZC_IU.mp4/Img2.jpg'
#input_image = caffe.io.load_image(example_image)
#_ = plt.imshow(input_image)
#prediction = gender_net.predict([input_image]) 

#print('predicted gender:', gender_list[prediction[0].argmax()])


# In[3]:

#prediction = age_net.predict([input_image]) 

#print('predicted age:', age_list[prediction[0].argmax()])


# ## Gender prediction

# In[7]:

#prediction = gender_net.predict([input_image]) 

#print('predicted gender:', gender_list[prediction[0].argmax()])


# In[12]:

# Haarcascade models

#eye_cascade = cv2.CascadeClassifier('/home/tonmoy/anaconda3/share/OpenCV/haarcascades/haarcascade_eye.xml')

#img = cv2.imread('test.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# In[6]:

#function to write in csv
import csv
def write_list_in_file(final, name):
    with open(name, "w", newline="",encoding="utf8") as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(final)


# In[7]:


import os
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]



# In[8]:

#Function to read csv files
from csv import reader
# Load a CSV file\n",
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[ ]:

import pylab
import imageio
import os

img_path = './images/talkshow/tritiyomatra/new' #path to images
csv_path = './csvs/final_csvs/tritiyomatra/new'
dir_list = get_immediate_subdirectories(img_path)
#print(dir_list)

for directory in dir_list:
   
    print(directory)
    path = csv_path+'/result1_'+directory+'.csv'
    if(os.path.isfile(path)):
        continue
    else:
        myTable = load_csv(csv_path+'/result_'+directory+'.csv')
        print(len(myTable))
        myTable[0].append("gender")
        for i in range(1, len(myTable)):
            try:  
                input_image = caffe.io.load_image(img_path+'/'+directory+'/Img'+str(i)+'.jpg')
                #predict gender
                #_ = plt.imshow(input_image)
                prediction = gender_net.predict([input_image]) 
                #print(img)
                #print('predicted gender:', gender_list[prediction[0].argmax()])
                if(i%100==0):
                    print(i);
                #prediction = gender_net.predict([input_image]) 
                myTable[i].append(str(gender_list[prediction[0].argmax()]))

                #i = i+1
            except:
                #i = i+1
                continue;
        #Write the results in csv 
        write_list_in_file(myTable,csv_path+'/result1_'+directory+'.csv')





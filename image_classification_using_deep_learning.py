#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Algoritmaları ile Fotoğraflardaki Nesneleri Tanıma ve Sınıflandırma Projesi
# 
# <IMG src="deep7.png" width="750" height="180">
#     
#     

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Datasetimizi (cifar10 verisetini) yüklüyoruz: (Yükleme işlemi için Internet bağlantınızın olması gerekiyor). Eğer bağlantınız yoksa  veri setini Internetten indirip de yükleyebilirsiniz..

# In[2]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()


# In[4]:


X_train.shape


# #### Her bir fotoğraf 32 pixele-32 pixel kare boyutunda ve renkli 3 kanal RGB bilgileri olduğu için arrayımız bu şekilde..
# 
# <IMG src="cifar10_images.jpg" width="400" height="400">
# 

# In[5]:


X_test.shape


# In[6]:


y_train[:3]


# y_train ve y_test 2 boyutlu bir array olarak tutuluyor cifar10 verisetinde. 
# Biz bu verileri görsel olarak daha rahat anlamak için tek boyutlu hale getiriyoruz.
# 2 boyutlu bir arrayi (sadece tekbir boyutunda veri var diğer boyutu boş olan tabi) tekboyutlu hale geitrmek için reshape() kullanıyoruz..

# In[7]:


y_test = y_test.reshape(-1,)


# In[8]:


y_test 


# #### Verilere bir göz atalım. bu amaçla kendimiz bir array oluşturuyoruz: 

# In[9]:


resim_siniflari = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[10]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])        
    plt.xlabel(resim_siniflari[y[index]])


# In[11]:


plot_sample(X_test, y_test, 0)


# In[12]:


plot_sample(X_test, y_test, 1)


# In[13]:


plot_sample(X_test, y_test, 3)


# ### Normalization
# 
# Verilerimizi normalize etmemiz gerekiyor. Aksi takdirde CNN algoritmaları yanlış sonuç verebiliyor. Fotoğraflar RGB olarak 3 kanal ve her bir pixel 0-255 arasında değer aldığı için normalization için basitçe her bir pixel değerini 255'e bölmemiz yeterli..

# In[14]:


X_train = X_train / 255
X_test = X_test / 255


# ### Deep Learning Algoritmamızı CNN - Convolutional Neural Network Kullanarak Tasarlıyoruz:

# In[15]:


deep_learning_model = models.Sequential([
    # İlk bölüm Convolution layer.. Bu kısımda fotoğraflardan tanımlama yapabilmek için özellikleri çıkarıyoruz...
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # İkinci bölüm klasik Articial Neural Network olan layerımız.. Yukarıdaki özelliklerimiz ve training bilgilerine
    # göre ANN modelimizi eğiteceğiz..
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[16]:


deep_learning_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### Modelimizi eğitmeye başlayalım artık...

# In[ ]:


deep_learning_model.fit(X_train, y_train, epochs=5)


# In[ ]:


deep_learning_model.evaluate(X_test,y_test)


# In[ ]:


y_pred = deep_learning_model.predict(X_test)
y_pred[:3]


# In[29]:


y_predictions_siniflari = [np.argmax(element) for element in y_pred]
y_predictions_siniflari[:3]


# In[30]:


y_test[:3]


# In[31]:


plot_sample(X_test, y_test,0)


# In[32]:


resim_siniflari[y_predictions_siniflari[0]]


# In[33]:


plot_sample(X_test, y_test,1)


# In[34]:


resim_siniflari[y_predictions_siniflari[1]]


# In[35]:


plot_sample(X_test, y_test,2)


# In[36]:


resim_siniflari[y_predictions_siniflari[2]]


# In[ ]:





# In[ ]:





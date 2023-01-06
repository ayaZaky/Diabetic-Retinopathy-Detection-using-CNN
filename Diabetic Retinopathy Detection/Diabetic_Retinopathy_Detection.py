import tkinter as tk
from tkinter import ttk
import sys
import os
from tkinter import *
from tkinter import messagebox,filedialog
import numpy as np
from PIL import Image, ImageTk ,ImageFilter
import cv2
import os
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.cluster import MeanShift, estimate_bandwidth
import decimal as d
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
 
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
 
import keras
import cv2 as ocv
from keras.preprocessing import image
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from numpy.random import rand, shuffle
from PIL import Image, ImageTk
from itertools import count, cycle
from tensorflow.keras import datasets, layers, models
         
 
class Application(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        self.pack()

        self.master.geometry("800x700")
        self.master.title("Image Processing ")
        self.master.configure(bg='gray12')
        self.create_widgets()
        self.flag=0 
         
    ## show processed img
    def show_new_image(self):
        self.image_bgr_resize = cv2.resize(self.new_image, self.new_size, interpolation=cv2.INTER_AREA)#
        self.image_bgr_resize = cv2.normalize(self.image_bgr_resize, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.image_PIL = Image.fromarray(self.image_bgr_resize) #Convert from RGB to PIL format
        self.image_tk2 = ImageTk.PhotoImage(self.image_PIL) #Convert to ImageTk format
        self.canvas1.create_image(450,240, image=self.image_tk2)
        self.canvas1.grid(column=0, row=1)

             #########################################################################################################################################################################
    def classification(self): 
        dir = 'Diabetic Retinopathy'
        classes = ['Mild','Moderate','No_DR','Proliferate_DR','Severe']
         

 
    ############
        file_path = 'cnn_model_2.h5'
         
        if os.path.exists(file_path):
             # using saved training model ..
            model=keras.models.load_model(file_path)
    
            # training will happen if the trained model isnot available
        else: 
            Data = []
            Lables = []
            for category in os.listdir(dir): 
                newPath = os.path.join(dir,category)
                for img in os.listdir(newPath):
                    img_path = os.path.join(newPath,img)
                    if 'Thumbs.db' not in img_path:
                        print(img_path)
                        #im = ocv.imread(img_path)
                        #b, g, r = cv2.split(im)
                        im = Image.open(img_path)
                        img = np.array(im.resize((224,224)))

                        # convert to green channel only
                        img[:,:,[0,2]] = 0
            
                        Data.append(img)
                        Lables.append(classes.index(category))
            
            from sklearn.utils import shuffle

            data,label = shuffle(Data,Lables, random_state=42)
            train_data = [data,label]

            x_train, x_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size = 0.1, random_state = 42)

            print(np.array(x_train).shape)
            print(np.array(y_train).shape)

            y_train = np_utils.to_categorical(np.array(y_train), 5)
            y_test = np_utils.to_categorical(np.array(y_test), 5)

            x_train = np.array(x_train).astype("float32")/255.
            x_test = np.array(x_test).astype("float32")/255.

            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train[0].shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax')) 

            # compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            file_path = 'cnn_model_2.h5'
            modelcheckpoint = ModelCheckpoint(file_path,monitor='val_loss',verbose=2,save_best_only=True,mode='min')
            callBackList = [modelcheckpoint]

            # Fit the model
            model=model.fit(x_train,y_train,epochs=12,validation_split=0.25,callbacks=callBackList)
     
     # Make prediction 
        self.new_image=self.filename
        im = Image.open(self.new_image)
        img = np.array(im.resize((224,224))) 
        # convert to green channel only
        img[:,:,[0,2]] = 0
        img=np.array(img).astype("float32")/255
        img = img.reshape(1,224,224,3)
        print(classes[(np.argmax(model.predict(img)))]) 
        predict = classes[(np.argmax(model.predict(img)))]
 
        #acc = model.evaluate(x_train,y_train) 
       # messagebox.showinfo("Model Prediction", "Prediction is: [ "+predict+" ]\n\n"+"Accurecy of model : "+ str(acc[1]*100)+"%"+"\n\nLoss of model: "+str(acc[0]))
         
        #######
        self.label_class= Label(self.frame3, font=('Roboto',15) ,background='Deep Sky Blue3',fg='gray20' ,relief='solid')
        self.label_class.grid( column=1, row= 0)
         
         
        
        self.label_class.config(text="Prediction = "+ predict)
        #self.label_acc.config(text="Accurecy= "+str(acc[1]*100)[:5]+"%")
        #self.label_loss.config(text="Loss = "+str(acc[0]*100)[:5]+"%")
         
        #self.myload =circularloadbar.CircularLoadBar(root, 360, 180, 200, 150)
        
      
      #########################################################################################################################################################################
     
    ### GUI #####
    def create_widgets(self):
        self.configure(bg='gray12')   

        #Canvas
        self.canvas1 = tk.Canvas(self)
        self.canvas1.configure(width=800,height=480, bg='gray15',highlightbackground="Deep Sky Blue3",highlightthickness=1) 
        self.canvas1.grid(column=0, row=1)
        #self.canvas1.grid(padx=20, pady=20)

        
       #frame1
        self.frame1_button = tk.Canvas(self)
        self.frame1_button.configure(width=640,height=50,background='gray20')#,highlightbackground="mediumpurple")
        self.frame1_button.grid(column=0,row=0)
        self.frame1_button.grid(padx=20, pady=30)
    
        self.label= Label(self.frame1_button, font=('Roboto',18), background='gray20',fg='Deep Sky Blue3',relief='solid',)
        self.label.grid( column=5, row= 0)
        
         
         
        
        self.label.config(text="Diabetic Retinopathy Detection using CNN ")
        #Diabetic Retinopathy Detection using CNN
         
      #Frame2
        self.frame2_button = tk.Canvas(self)
        self.frame2_button.configure(width=640,height=50,background='gray20')
        self.frame2_button.grid(column=0,row=2)
        self.frame2_button.grid(padx=20, pady=30) 

       
        #frame3
        self.frame3 = tk.Canvas(self)
        self.frame3.configure(width=640,height=50,background='gray20')#,highlightbackground="mediumpurple")
        self.frame3.grid(column=0,row=3)
        self.frame3.grid(padx=20, pady=30)
    

        #File open and Load Image
        self.new_size=(40,40)
        self.img3 = Image.open('new.png')
        self.imag_resize3 = self.img3.resize(self.new_size)
        self.img_3 = ImageTk.PhotoImage(self.imag_resize3)
        self.button_open = tk.Button(self.frame2_button,image=self.img_3,compound = LEFT,activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=50 ,width=200)
        self.button_open.configure(text = '    New Image  ' )
        self.button_open.grid(column=0, row=1)
        self.button_open.configure(command=self.loadImage) 

        #classification Button 
         
        self.new_size=(30,30)
        self.img4 = Image.open('classification.png')
        self.imag_resize4 = self.img4.resize(self.new_size)
        self.img_4 = ImageTk.PhotoImage(self.imag_resize4)
        self.button_cls = tk.Button(self.frame2_button,image=self.img_4,compound = LEFT,text='Classification',activebackground='Deep Sky Blue3',activeforeground='gray18',background='gray20',fg='Deep Sky Blue3',font=('Roboto',15),relief='solid',height=50 ,width=200)
        self.button_cls.config( text='  Classifiy   ')
        self.button_cls.grid( column=1, row=1 )
        self.button_cls.configure(command = self.classification)

        ###############################
       
         
     ##### Event Call Back###

    def loadImage(self): 
        self.filename= filedialog.askopenfilename() 
        self.image_bgr = cv2.imread(self.filename)
         
        self.height, self.width = self.image_bgr.shape[:2]
        print(self.height, self.width)
        if self.width > self.height:
            self.new_size = (700,400)
        else:
            self.new_size = (700,400) 
             

        self.image_bgr_resize = cv2.resize(self.image_bgr, self.new_size, interpolation=cv2.INTER_AREA)
        self.image_bgr_resize = cv2.normalize(self.image_bgr_resize, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
        #self.image_rgb = cv2.cvtColor(self.image_bgr_resize, cv2.COLOR_BGR2GRAY) #Since imread is BGR, it is converted to RGB
        self.image_PIL = Image.fromarray(self.image_bgr_resize) #Convert from RGB to PIL format
        self.image_tk = ImageTk.PhotoImage(self.image_PIL) #Convert to ImageTk format
        self.canvas1.create_image(400,240, image=self.image_tk)

 

    def quit_app(self):
        self.Msgbox = tk.messagebox.askquestion("Exit Applictaion", "Are you sure?", icon="warning")
        if self.Msgbox == "yes":
            self.master.destroy()
        
            
 
        

def main():
    root = tk.Tk()
    app = Application(master=root)#Inherit 
    app.mainloop()
if __name__ == "__main__":
    main()


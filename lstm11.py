# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:49:49 2019

@author: Shalini Bhawsingka
"""
import matplotlib.pyplot
import numpy as np
import pandas as pd  
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk
root = tk.Tk()
from tkinter import messagebox
from tkinter import ttk

def main():
    def callbackFunc(event):
        x = comboExample.get()
        if x == "HCL":
            hcl()
        elif x == "OFFS":
            offs()
        elif x == "TCS":
            tcs()
        elif x =="WIPRO":
            wipro()
        elif x =="MINDTREE":
            mindtree()
        else:
                techm()
    
    #CODE TO TRAIN HCL
    def hcl():
        def train():
            root2 = tk.Toplevel()
            root2.title("Graph")
            
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(history.history['loss'])
            a.set_title('Stock Price Predicted')
            a.set_xlabel('epochs', fontsize=14)
            a.set_ylabel('loss of error', fontsize=14)
            a.legend(['train', 'test'], loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            canvas = FigureCanvasTkAgg(f, root2)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, root2)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            root2.mainloop()
            
        def pred():
            root1 = tk.Toplevel()
            root1.title("Graph")
            
            x = real_stock_price[:100,]
            y = predicted_stock_price[:100,]
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(x, color = 'red', label = 'Real Strock Price')
            a.plot(y, color = 'blue', label = 'Predicted Strock Price')
            
            canvas = FigureCanvasTkAgg(f, root1)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            a.set_title('Stock Price Predicted')
            
            toolbar = NavigationToolbar2TkAgg(canvas, root1)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            a.set_xlabel('Time', fontsize=14)
            a.set_ylabel('Stock Price', fontsize=14)
            a.legend(loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            """plt.plot(x, color = 'red', label = 'Real Strock Price')
            plt.xlabel('Time')
            plt.plot(y, color = 'blue', label = 'Predicted Strock Price')
            plt.title('Stock Price Predicted')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()"""
            root1.mainloop()
            
        
            
        dataset_train = pd.read_csv('NSE-HCL_INSYS.csv')
        training_set = dataset_train.iloc[:, 2:9].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        """training_set_scaled = sc.fit_transform(training_set)"""
        
        from sklearn.model_selection import train_test_split
        training_set, testing_set = train_test_split(training_set, test_size = 0.3, random_state = 1)
        
        training_set_close = dataset_train.iloc[:, 6:7].values
        
        train_right = []
        y_train = []
        for i in range(60, 1389):
            train_right.append(training_set_close[i-60:i, 0])
            y_train.append(training_set_close[i, 0])
        train_right, y_train = np.array(train_right), np.array(y_train)
        
        train_left = dataset_train.iloc[:1329, [2,3,4,5,7,8]].values
        X_train = np.column_stack((train_left, train_right))
        X_train = sc.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        y_train = sc.fit_transform(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        
        regressor = Sequential()
        
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)
        
        real_stock_price = testing_set[:, 4:5]
        
        X_test_left = testing_set[:536, [0,1,2,3,5,6]]
        X_test_right = []
        inputs = sc.fit_transform(real_stock_price)
        for i in range(60, 596):
            X_test_right.append(inputs[i-60:i, 0])
        X_test_right = np.array(X_test_right)
        X_test = np.column_stack((X_test_left, X_test_right))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        button = tk.Button(root, 
                           text ="Check the loss while the model is being trained", 
                           background = "floral white",
                           command = train)
        button.pack(pady = 25, padx = (120,0), side = 'left', fill = 'x')
        
        button = tk.Button(root, 
                           text ="Click to see the test once again", 
                           background = "floral white",
                           command = pred)
        button.pack(pady = 25, padx = (0,150), side = 'right', fill = 'x')
        
        result = tk.messagebox.askyesno("Message", "Training Done!! Do want to see the Graph")
        if(result == 1):
            pred()
        
    #CODE TO TRAIN OFFS
    def offs():
        def train():
            root2 = tk.Toplevel()
            root2.title("Graph")
            
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(history.history['loss'])
            a.set_title('Stock Price Predicted')
            a.set_xlabel('epochs', fontsize=14)
            a.set_ylabel('loss of error', fontsize=14)
            a.legend(['train', 'test'], loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            canvas = FigureCanvasTkAgg(f, root2)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, root2)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            root2.mainloop()
            
        def pred():
            root1 = tk.Toplevel()
            root1.title("Graph")
            
            x = real_stock_price[:100,]
            y = predicted_stock_price[:100,]
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(x, color = 'red', label = 'Real Strock Price')
            a.plot(y, color = 'blue', label = 'Predicted Strock Price')
            
            canvas = FigureCanvasTkAgg(f, root1)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            a.set_title('Stock Price Predicted')
            
            toolbar = NavigationToolbar2TkAgg(canvas, root1)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            a.set_xlabel('Time', fontsize=14)
            a.set_ylabel('Stock Price', fontsize=14)
            a.legend(loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            """plt.plot(x, color = 'red', label = 'Real Strock Price')
            plt.xlabel('Time')
            plt.plot(y, color = 'blue', label = 'Predicted Strock Price')
            plt.title('Stock Price Predicted')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()"""
            root1.mainloop()
            
        
            
        dataset_train = pd.read_csv('NSE-OFSS.csv')
        training_set = dataset_train.iloc[:, 2:9].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        """training_set_scaled = sc.fit_transform(training_set)"""
        
        from sklearn.model_selection import train_test_split
        training_set, testing_set = train_test_split(training_set, test_size = 0.3, random_state = 1)
        
        training_set_close = dataset_train.iloc[:, 6:7].values
        
        train_right = []
        y_train = []
        for i in range(60, 1389):
            train_right.append(training_set_close[i-60:i, 0])
            y_train.append(training_set_close[i, 0])
        train_right, y_train = np.array(train_right), np.array(y_train)
        
        train_left = dataset_train.iloc[:1329, [2,3,4,5,7,8]].values
        X_train = np.column_stack((train_left, train_right))
        X_train = sc.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        y_train = sc.fit_transform(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        
        regressor = Sequential()
        
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        real_stock_price = testing_set[:, 4:5]
        
        X_test_left = testing_set[:536, [0,1,2,3,5,6]]
        X_test_right = []
        inputs = sc.fit_transform(real_stock_price)
        for i in range(60, 596):
            X_test_right.append(inputs[i-60:i, 0])
        X_test_right = np.array(X_test_right)
        X_test = np.column_stack((X_test_left, X_test_right))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        button = tk.Button(root, 
                           text ="Check the loss while the model is being trained", 
                           background = "floral white",
                           command = train)
        button.pack(pady = 25, padx = (120,0), side = 'left', fill = 'x')
        
        button = tk.Button(root, 
                           text ="Click to see the test once again", 
                           background = "floral white",
                           command = pred)
        button.pack(pady = 25, padx = (0,150), side = 'right', fill = 'x')
        
        result = tk.messagebox.askyesno("Message", "Training Done!! Do want to see the Graph")
        if(result == 1):
            pred()
    
    #CODE TO TRAIN TCS
    def tcs():
        def train():
            root2 = tk.Toplevel()
            root2.title("Graph")
            
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(history.history['loss'])
            a.set_title('Stock Price Predicted')
            a.set_xlabel('epochs', fontsize=14)
            a.set_ylabel('loss of error', fontsize=14)
            a.legend(['train', 'test'], loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            canvas = FigureCanvasTkAgg(f, root2)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, root2)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            root2.mainloop()
            
        def pred():
            root1 = tk.Toplevel()
            root1.title("Graph")
            
            x = real_stock_price[:100,]
            y = predicted_stock_price[:100,]
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(x, color = 'red', label = 'Real Strock Price')
            a.plot(y, color = 'blue', label = 'Predicted Strock Price')
            
            canvas = FigureCanvasTkAgg(f, root1)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            a.set_title('Stock Price Predicted')
            
            toolbar = NavigationToolbar2TkAgg(canvas, root1)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            a.set_xlabel('Time', fontsize=14)
            a.set_ylabel('Stock Price', fontsize=14)
            a.legend(loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            """plt.plot(x, color = 'red', label = 'Real Strock Price')
            plt.xlabel('Time')
            plt.plot(y, color = 'blue', label = 'Predicted Strock Price')
            plt.title('Stock Price Predicted')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()"""
            root1.mainloop()
            
        
            
        dataset_train = pd.read_csv('NSE-TCS.csv')
        training_set = dataset_train.iloc[:, 2:9].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        """training_set_scaled = sc.fit_transform(training_set)"""
        
        from sklearn.model_selection import train_test_split
        training_set, testing_set = train_test_split(training_set, test_size = 0.3, random_state = 1)
        
        training_set_close = dataset_train.iloc[:, 6:7].values
        
        train_right = []
        y_train = []
        for i in range(60, 1389):
            train_right.append(training_set_close[i-60:i, 0])
            y_train.append(training_set_close[i, 0])
        train_right, y_train = np.array(train_right), np.array(y_train)
        
        train_left = dataset_train.iloc[:1329, [2,3,4,5,7,8]].values
        X_train = np.column_stack((train_left, train_right))
        X_train = sc.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        y_train = sc.fit_transform(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        
        regressor = Sequential()
        
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        real_stock_price = testing_set[:, 4:5]
        
        X_test_left = testing_set[:536, [0,1,2,3,5,6]]
        X_test_right = []
        inputs = sc.fit_transform(real_stock_price)
        for i in range(60, 596):
            X_test_right.append(inputs[i-60:i, 0])
        X_test_right = np.array(X_test_right)
        X_test = np.column_stack((X_test_left, X_test_right))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        button = tk.Button(root, 
                           text ="Check the loss while the model is being trained", 
                           background = "floral white",
                           command = train)
        button.pack(pady = 25, padx = (120,0), side = 'left', fill = 'x')
        
        button = tk.Button(root, 
                           text ="Click to see the test once again", 
                           background = "floral white",
                           command = pred)
        button.pack(pady = 25, padx = (0,150), side = 'right', fill = 'x')
        
        result = tk.messagebox.askyesno("Message", "Training Done!! Do want to see the Graph")
        if(result == 1):
            pred()
    
    #CODE TO TRAIN WIPRO
    def wipro():
        def train():
            root2 = tk.Toplevel()
            root2.title("Graph")
            
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(history.history['loss'])
            a.set_title('Stock Price Predicted')
            a.set_xlabel('epochs', fontsize=14)
            a.set_ylabel('loss of error', fontsize=14)
            a.legend(['train', 'test'], loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            canvas = FigureCanvasTkAgg(f, root2)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, root2)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            root2.mainloop()
            
        def pred():
            root1 = tk.Toplevel()
            root1.title("Graph")
            
            x = real_stock_price[:100,]
            y = predicted_stock_price[:100,]
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(x, color = 'red', label = 'Real Strock Price')
            a.plot(y, color = 'blue', label = 'Predicted Strock Price')
            
            canvas = FigureCanvasTkAgg(f, root1)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            a.set_title('Stock Price Predicted')
            
            toolbar = NavigationToolbar2TkAgg(canvas, root1)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            a.set_xlabel('Time', fontsize=14)
            a.set_ylabel('Stock Price', fontsize=14)
            a.legend(loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            """plt.plot(x, color = 'red', label = 'Real Strock Price')
            plt.xlabel('Time')
            plt.plot(y, color = 'blue', label = 'Predicted Strock Price')
            plt.title('Stock Price Predicted')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()"""
            root1.mainloop()
            
        
            
        dataset_train = pd.read_csv('NSE-WIPRO.csv')
        training_set = dataset_train.iloc[:, 2:9].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        """training_set_scaled = sc.fit_transform(training_set)"""
        
        from sklearn.model_selection import train_test_split
        training_set, testing_set = train_test_split(training_set, test_size = 0.3, random_state = 1)
        
        training_set_close = dataset_train.iloc[:, 6:7].values
        
        train_right = []
        y_train = []
        for i in range(60, 1389):
            train_right.append(training_set_close[i-60:i, 0])
            y_train.append(training_set_close[i, 0])
        train_right, y_train = np.array(train_right), np.array(y_train)
        
        train_left = dataset_train.iloc[:1329, [2,3,4,5,7,8]].values
        X_train = np.column_stack((train_left, train_right))
        X_train = sc.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        y_train = sc.fit_transform(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        
        regressor = Sequential()
        
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        real_stock_price = testing_set[:, 4:5]
        
        X_test_left = testing_set[:536, [0,1,2,3,5,6]]
        X_test_right = []
        inputs = sc.fit_transform(real_stock_price)
        for i in range(60, 596):
            X_test_right.append(inputs[i-60:i, 0])
        X_test_right = np.array(X_test_right)
        X_test = np.column_stack((X_test_left, X_test_right))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        button = tk.Button(root, 
                           text ="Check the loss while the model is being trained", 
                           background = "floral white",
                           command = train)
        button.pack(pady = 25, padx = (120,0), side = 'left', fill = 'x')
        
        button = tk.Button(root, 
                           text ="Click to see the test once again", 
                           background = "floral white",
                           command = pred)
        button.pack(pady = 25, padx = (0,150), side = 'right', fill = 'x')
        
        result = tk.messagebox.askyesno("Message", "Training Done!! Do want to see the Graph")
        if(result == 1):
            pred()
        #CODE TO TRAIN MINDTREE
          
    def mindtree():
        def train():
            root2 = tk.Toplevel()
            root2.title("Graph")
            
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(history.history['loss'])
            a.set_title('Stock Price Predicted')
            a.set_xlabel('epochs', fontsize=14)
            a.set_ylabel('loss of error', fontsize=14)
            a.legend(['train', 'test'], loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            canvas = FigureCanvasTkAgg(f, root2)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, root2)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            root2.mainloop()
            
        def pred():
            root1 = tk.Toplevel()
            root1.title("Graph")
            
            x = real_stock_price[:100,]
            y = predicted_stock_price[:100,]
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(x, color = 'red', label = 'Real Strock Price')
            a.plot(y, color = 'blue', label = 'Predicted Strock Price')
            
            canvas = FigureCanvasTkAgg(f, root1)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            a.set_title('Stock Price Predicted')
            
            toolbar = NavigationToolbar2TkAgg(canvas, root1)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            a.set_xlabel('Time', fontsize=14)
            a.set_ylabel('Stock Price', fontsize=14)
            a.legend(loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            """plt.plot(x, color = 'red', label = 'Real Strock Price')
            plt.xlabel('Time')
            plt.plot(y, color = 'blue', label = 'Predicted Strock Price')
            plt.title('Stock Price Predicted')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()"""
            root1.mainloop()
            
        
            
        dataset_train = pd.read_csv('NSE-MINDTREE.csv')
        training_set = dataset_train.iloc[:, 2:9].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        """training_set_scaled = sc.fit_transform(training_set)"""
        
        from sklearn.model_selection import train_test_split
        training_set, testing_set = train_test_split(training_set, test_size = 0.3, random_state = 1)
        
        training_set_close = dataset_train.iloc[:, 6:7].values
        
        train_right = []
        y_train = []
        for i in range(60, 1389):
            train_right.append(training_set_close[i-60:i, 0])
            y_train.append(training_set_close[i, 0])
        train_right, y_train = np.array(train_right), np.array(y_train)
        
        train_left = dataset_train.iloc[:1329, [2,3,4,5,7,8]].values
        X_train = np.column_stack((train_left, train_right))
        X_train = sc.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        y_train = sc.fit_transform(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        
        regressor = Sequential()
        
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        real_stock_price = testing_set[:, 4:5]
        
        X_test_left = testing_set[:536, [0,1,2,3,5,6]]
        X_test_right = []
        inputs = sc.fit_transform(real_stock_price)
        for i in range(60, 596):
            X_test_right.append(inputs[i-60:i, 0])
        X_test_right = np.array(X_test_right)
        X_test = np.column_stack((X_test_left, X_test_right))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        button = tk.Button(root, 
                           text ="Check the loss while the model is being trained", 
                           background = "floral white",
                           command = train)
        button.pack(pady = 25, padx = (120,0), side = 'left', fill = 'x')
        
        button = tk.Button(root, 
                           text ="Click to see the test once again", 
                           background = "floral white",
                           command = pred)
        button.pack(pady = 25, padx = (0,150), side = 'right', fill = 'x')
        
        result = tk.messagebox.askyesno("Message", "Training Done!! Do want to see the Graph")
        if(result == 1):
            pred()
            
            #CODE TO TRAIN TECHM
    def techm():
        def train():
            root2 = tk.Toplevel()
            root2.title("Graph")
            
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(history.history['loss'])
            a.set_title('Stock Price Predicted')
            a.set_xlabel('epochs', fontsize=14)
            a.set_ylabel('loss of error', fontsize=14)
            a.legend(['train', 'test'], loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            canvas = FigureCanvasTkAgg(f, root2)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2TkAgg(canvas, root2)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            root2.mainloop()
            
        def pred():
            root1 = tk.Toplevel()
            root1.title("Graph")
            
            x = real_stock_price[:80,]
            y = predicted_stock_price[:80,]
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111)
            a.plot(x, color = 'red', label = 'Real Strock Price')
            a.plot(y, color = 'blue', label = 'Predicted Strock Price')
            
            canvas = FigureCanvasTkAgg(f, root1)
            canvas.show()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            a.set_title('Stock Price Predicted')
            
            toolbar = NavigationToolbar2TkAgg(canvas, root1)
            toolbar.update()
            canvas.tkcanvas.pack(side=tk.TOP, width=800, height=500, fill=tk.BOTH, expand=True)
            
            a.set_xlabel('Time', fontsize=14)
            a.set_ylabel('Stock Price', fontsize=14)
            a.legend(loc='upper center', ncol=3, fancybox=True, shadow=True)
            
            """plt.plot(x, color = 'red', label = 'Real Strock Price')
            plt.xlabel('Time')
            plt.plot(y, color = 'blue', label = 'Predicted Strock Price')
            plt.title('Stock Price Predicted')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()"""
            root1.mainloop()
            
        
            
        dataset_train = pd.read_csv('NSE-TECHM.csv')
        training_set = dataset_train.iloc[:, 2:9].values
        
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0, 1))
        """training_set_scaled = sc.fit_transform(training_set)"""
        
        from sklearn.model_selection import train_test_split
        training_set, testing_set = train_test_split(training_set, test_size = 0.3, random_state = 1)
        
        training_set_close = dataset_train.iloc[:, 6:7].values
        
        train_right = []
        y_train = []
        for i in range(60, 1389):
            train_right.append(training_set_close[i-60:i, 0])
            y_train.append(training_set_close[i, 0])
        train_right, y_train = np.array(train_right), np.array(y_train)
        
        train_left = dataset_train.iloc[:1329, [2,3,4,5,7,8]].values
        X_train = np.column_stack((train_left, train_right))
        X_train = sc.fit_transform(X_train)
        y_train = y_train.reshape(-1, 1)
        y_train = sc.fit_transform(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        
        regressor = Sequential()
        
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        real_stock_price = testing_set[:, 4:5]
        
        X_test_left = testing_set[:536, [0,1,2,3,5,6]]
        X_test_right = []
        inputs = sc.fit_transform(real_stock_price)
        for i in range(60, 596):
            X_test_right.append(inputs[i-60:i, 0])
        X_test_right = np.array(X_test_right)
        X_test = np.column_stack((X_test_left, X_test_right))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = regressor.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
        button = tk.Button(root, 
                           text ="Check the loss while the model is being trained", 
                           background = "floral white",
                           command = train)
        button.pack(pady = 25, padx = (120,0), side = 'left', fill = 'x')
        
        button = tk.Button(root, 
                           text ="Click to see the test once again", 
                           background = "floral white",
                           command = pred)
        button.pack(pady = 25, padx = (0,150), side = 'right', fill = 'x')
        
        result = tk.messagebox.askyesno("Message", "Training Done!! Do want to see the Graph")
        if(result == 1):
            pred()
    
    #CODE COMMON FOR EVERY COMPANY TO SELECT THE RIGHT DATASET
    tk.Label(root, 
		 text="Welcome to Stock Prediction",
		 fg = "gray2",
		 bg = "floral white",
		 font = "Helvetica 16 bold italic").pack()
    
    tk.Label(root, 
		 text="Please select the company's dataset and then see the loss by clicking the train button",
		 fg = "gray2",
		 bg = "floral white",
		 font = "Helvetica 16 bold italic").pack()
    
    labelTop = tk.Label(root,
                    text = "Choose the company", fg = "gray45", bg = "floral white", font = "Helvetica 16 bold italic")
    labelTop.pack(padx = 10, pady = 25)

    comboExample = ttk.Combobox(root,   
                            values=[
                                    "HCL", 
                                    "OFFS",
                                    "TCS",
                                    "WIPRO",
                                    "MINDTREE",
                                    "TECH_MAHINDRA"])


    comboExample.pack(padx = 50, pady = 5)
    comboExample.current(1)
    combostyle = ttk.Style()

    combostyle.theme_create('combostyle', parent='alt',
                         settings = {'TCombobox':
                                     {'configure':
                                      {'selectbackground': 'gray1',
                                       'fieldbackground': 'old lace',
                                       'background': 'snow3'
                                       }}}
                         )
    combostyle.theme_use('combostyle') 
    
    comboExample.bind("<<ComboboxSelected>>", callbackFunc)
    
    status = tk.Label(root, text = "Please wait after selecting the dataset...A popup will appear to proceed further...", bd = 1, relief = 'sunken', anchor = 'w')
    status.pack(side = 'bottom', fill = 'x', pady = (20,2))
    
    root.configure(background="lavender")
    root.title("Stock Prediction") 
    root.mainloop()
        
main()

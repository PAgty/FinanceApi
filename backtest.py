import collections
from itertools import islice
import numpy as np
import json

class backtest:
    def __init__(self,model):
        has_fit = hasattr(model, 'fit') and callable(getattr(model, 'fit'))
        has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
        if has_fit and has_predict:
            self.model = model
        else:
            raise AttributeError('given model must has fit and predict method at the same time')
            
    def test(self,data,window_size = None,forcasting_size = None,sliding_steps = None):
        if not isinstance(data,dict):
            raise TypeError('given data can only be dict at this point')
        size = len(data)
        self.window_size = window_size
        self.forcasting_size = forcasting_size
        self.sliding_steps = sliding_steps
        self.check_size(size,window_size,forcasting_size,sliding_steps)
        anchor = 0
        tail = anchor + self.window_size + self.forcasting_size
        number_of_testing = 0
        self.prediction_error = []
        while tail < size:
            sliced = islice(data.items(), anchor, anchor + self.window_size)
            training_data = collections.OrderedDict(sliced)
            self.model.fit(training_data)
            predicted_value = self.model.predict(time = self.forcasting_size)
            values = list(data.values())
            y_truth = values[anchor + self.window_size:anchor + self.window_size + self.forcasting_size]

#             print(y_truth)
            error = self.mean_absolute_percentage_error(y_truth,predicted_value)
            self.prediction_error.append(error)
            anchor = anchor + self.sliding_steps
            tail = anchor + self.window_size + self.forcasting_size
            number_of_testing += 1
#             if number_of_testing == 7:
#                 print('the {} training length:{}'.format(len(training_data),training_data))
#                 print('the {} truth length:{}'.format(len(y_truth),y_truth))
#                 print('the {} predict length:{}'.format(len(predicted_value),predicted_value))
            print(str(number_of_testing) + " number of testing : " + 'error ' + str(error))
        return self.prediction_error,number_of_testing
            
    def check_size(self,size,window_size,forcasting_size,sliding_steps):
        if window_size == None or window_size >= size:
            print('--------window size is not provided or too large,reset to 1/3 of data length--------')
            self.window_size = int(size/3)
        if forcasting_size == None or forcasting_size >= self.window_size:
            print('---------forcasting size is not provided or too large,reset to 1/3 of training window length-------')
            self.forcasting_size = int(self.window_size/3)
        if sliding_steps == None or sliding_steps >= self.forcasting_size:
            print('----------sliding steps is not provided or too large,reset to 2/3 of forcasting window length-------')
            self.sliding_steps = int(2 * self.forcasting_size/3)
            
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        '''
            not sure about zero division
        '''
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

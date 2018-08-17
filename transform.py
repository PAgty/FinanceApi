import abc
import collections

class transform(abc.ABC):
    data = {}
    params = {}
    time_mapping = {'S': 1, 'M': 60, 'H': 1440, 'D':34560}
    @abc.abstractmethod
    def __init__(self, params):
        self.params = params
        
    @abc.abstractmethod
    def fit(self,data):      
        self.data = data

    @abc.abstractmethod
    def apply(self,date_period):
        time = date_period
        
    def convert(self,x):
        if x.isdigit(): 
            return int(x)
        else:
            try:
                val = float(x)
                return val
            except:
                return 


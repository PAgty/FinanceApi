import abc
import collections

class algorithm(abc.ABC):
	data = {}
	@abc.abstractmethod
	def __init__(self, params):
		self.params = params

	@abc.abstractmethod
	def fit(self,data):
		self.data = data

	@abc.abstractmethod
	def predict(self):
		print(predicting)

	def preprocess(self,data):
		od = collections.OrderedDict(sorted(data.items()))
		return od

	def convert(self,x):
		if x.isdigit(): 
			return int(x)
		else:
			try:
				val = float(x)
				return val
			except:
				return 
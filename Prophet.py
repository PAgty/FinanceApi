from utility import infer_freq
import pandas as pd
from algorithms import algorithm
from fbprophet import Prophet as fbProphet

class Prophet(algorithm):
	def __init__(self,params = {}):
		if params:
			for k,v in params.items():
				setattr(self,k,int(v))
		self.model = fbProphet()

	def fit(self,data):
		self.data = pd.Series(data)
		self.data.index = pd.to_datetime(self.data.index,format = '%Y-%m-%d')
		self.data = pd.to_numeric(self.data, errors='coerce')
		self.data = self.data.astype(float)
		self.df = pd.DataFrame({'ds':self.data.index, 'y':self.data.values})
		self.model.fit(self.df)

	def predict(self,time = 5):
		self.freq = infer_freq(self.data)
		future = self.model.make_future_dataframe(periods=time,freq =self.freq)
		forecast = self.model.predict(future)
		self.time_dict = self._get_prediction_with_history(forecast)
		return self.time_dict

	def _get_prediction_with_history(self,future):
		df = future[['ds','yhat']]
		data_dict = dict(zip(df.ds,df.yhat))
		time_dict = dict((k.strftime("%Y-%m-%d"),v) for (k,v) in data_dict.items())
		return time_dict

# data = {"2004-09-01": "14089", "2005-09-01": "5758", "2006-03-01": "3382", "2006-06-01": "3186", "2006-09-01": "3271", "2007-01-01": "5300", "2007-03-01": "20518", "2007-06-01": "18373", "2007-09-01": "53265", "2008-01-01": "91193", "2008-03-01": "223883", "2008-06-01": "197119", "2008-09-01": "237820", "2009-01-01": "208096", "2009-03-01": "100704", "2009-06-01": "92290", "2009-09-01": "60550", "2010-01-01": "44254", "2010-03-01": "110052", "2010-06-01": "157617", "2010-09-01": "117208", "2011-01-01": "92753", "2011-03-01": "64638", "2011-06-01": "154511", "2011-09-01": "151503", "2012-01-01": "136955", "2012-03-01": "176194", "2012-06-01": "196928", "2012-09-01": "194400", "2013-01-01": "240000", "2013-03-01": "296500", "2013-06-01": "306500", "2013-09-01": "385300", "2014-01-01": "402300", "2014-03-01": "309200", "2014-06-01": "387100", "2014-09-01": "330300", "2015-01-01": "438700", "2015-03-01": "415600", "2015-06-01": "541800", "2015-09-01": "731900", "2016-01-01": "1074900", "2016-03-01": "1010500", "2016-06-01": "1649000", "2016-09-01": "1309100", "2017-01-01": "810600", "2017-03-01": "986800", "2017-06-01": "2362600", "2017-09-01": "3003400", "2018-01-01": "5212800"}
# m = Prophet()
# m.fit(data)
# ans = m.predict(20)
# print(ans)
# dict((k.strftime("%Y-%m-%d %H:%M:%S"),v)for (k,v) in time_dict.items())
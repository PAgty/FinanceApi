from algorithms import algorithm
from datetime import datetime
from transform import transform
import pandas as pd
import numpy as np
from woxiaxiede_avg import woxiaxiede_avg
from utility import infer_freq
import re


class woxiaxiede_agg(transform):
	diff = {}
	df = {}
	converted_df = {}
	data_freq = ""

	def __init__(self, weight=[]):
		if weight:
			self.weight = weight

	def fit(self, data, freq=None):
		# self.time = data.keys()
		self.df = pd.Series(data)
		self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
		self.df = pd.to_numeric(self.df, errors='coerce')
		self.df = self.df.astype(float)
		if not freq:
			freq = infer_freq(self.df)
		self.data_freq = freq
		self.df.index.freq = self.data_freq

	def apply(self, targetStartDate='', targetEndDate='', targetFreq='D', weight=[]):
		if weight:
			self.weight = weight
		else:
			self.weight = self.compute_weight(self.df.index)
			self.avg_df = self.df[0:-1]/self.weight
			self.avg_df = self.append_value(self.avg_df, self.weight)
			model = woxiaxiede_avg()
			model.fit(self.avg_df, self.df.index.freq)
			self.dataDict = model.apply(
				targetStartTime=targetStartDate, targetEndTime=targetEndDate, targetFreq=targetFreq)
			return self.dataDict

	def adjust(self, df, old_df):
		index = 0
		old_index = 0
		# print(df)
		for i in self.weight:
			stride = i/2
			if i % 2 == 0:
				adj_1 = df.iloc[int(index + stride)] - old_df.iloc[int(old_index)]
				adj_2 = df.iloc[int(index + stride - 1)] - old_df.iloc[int(old_index)]
				adj = (adj_1 + adj_2)/2
			else:
				adj = df.iloc[int(index + stride)] - old_df.iloc[int(old_index)]
			df.iloc[int(index):int(index + i)] = df.iloc[int(index):int(index + i)] - adj
			index = index + i
			old_index = old_index + 1
		return df

	def compute_weight(self, index):
		diff = []
		self.df = self.append_index(self.df)
		index_length = len(self.df.index)
		# print(self.df.index)
		for i in range(1, index_length):
			value = (self.df.index[i] - self.df.index[i - 1]).total_seconds()
			diff.append(value/86400)
			# else:
			# 	if final_freq == 'M':
			# 		diff.append(self.diff_month(self.df.index[i],self.df.index[i - 1]))
		return diff

	def append_index(self, df):
		# index_length = len(df)
		# print(index)
		# print(df)
		if df.index.freq:
				# time_stamp = index[index_length - 1] + relativedelta(months = 3)
				# last = df.iloc[-1]
				# next_last = df.iloc[-2]
				# value = ((last - next_last)/weight[-2]) * weight[-1]
				# print(next_last)
				# print(last)
				# print(weight)
				# print(value)
				df[df.index[-1] + 1] = np.nan
				# print(self.df)
				# print(time_stamp.date())
				# self.df.set_value(pd.DatetimeIndex([time_stamp.date()]),np.nan)
				# index.append(pd.DatetimeIndex([time_stamp.date()]))

		return df

	def append_value(self, df, weight):
		if df.index.freq:
			# time_stamp = index[index_length - 1] + relativedelta(months = 3)
			last = df.iloc[-1]
			next_last = df.iloc[-2]
			value = ((last - next_last)/weight[-2]) * weight[-1]
			# print(next_last)
			# print(last)
			# print(weight)
			# print(value)
			df[df.index[-1] + 1] = value
		return df

	def diff_month(self, d1, d2):
		return (d1.year - d2.year) * 12 + d1.month - d2.month

	# def result(self,converted_df,startdate,enddate):
	# 	# if time == 'S':
	# 	# 	return converted_df
	# 	print(converted_df)
	# 	if self.time == 'M':
	# 		return converted_df.resample('1M').agg(np.sum)
	# 	if self.time == 'W':
	# 		return converted_df.resample('W').agg(np.sum)
	# 	# if time == 'H':
	# 	# 	return converted_df.resample('H').agg(np.sum)
	# 	if self.time == 'QS':
	# 		return converted_df.resample('3M').agg(np.sum)
	# 	if self.time == 'D':
	# 		return converted_df

# data = {"2006-01-01": "1884", "2006-04-01": "3646", "2006-07-01": "5757", "2006-10-01": "8200", "2007-01-01": "2415", "2007-04-01": "4429", "2007-07-01": "6272", "2007-10-01": "8423", "2008-01-01": "1987", "2008-04-01": "4351", "2008-07-01": "6243", "2008-10-01": "9184", "2009-01-01": "2251", "2009-04-01": "4966", "2009-07-01": "6782", "2009-10-01": "9606"}
# model = woxiaxiede_agg()
# model.fit(data)
# # conv_model = model.predict('D')
# # print(conv_model.resample('1M').sum())
# # s = 0
# # for i in range(1,31):
# # 	s = s + conv_model['2004-09-%.2d'%i]
# # print(s)
# time = 'D'
# conv = model.apply(time = time)
# print(conv)
# print(sum(conv[0:91]))

		# else:
		# 	if re.match('^D',self.user_freq):
		# 		time_stamp = index[index_length - 1] + relativedelta(days = 1)
		# 		index.append(time_stamp.date())
		# 	if re.match('^H',self.user_freq):
		# 		time_stamp =index[index_length - 1] + relativedelta(hours = 1)
		# 		index.append(time_stamp.date())
		# 	if re.match('^min',self.user_freq):
		# 		time_stamp = index[index_length - 1] + relativedelta(minutes = 1)
		# 		index.append(time_stamp)
		# 	if re.match('^S',self.user_freq):
		# 		time_stamp = index[index_length - 1] + relativedelta(seconds = 1)
		# 		index.append(time_stamp)
		# 	if re.match('^M',self.user_freq):
		# 		time_stamp = index[index_length - 1] + relativedelta(months = 1)
		# 		index.append(time_stamp)
		# 	if re.match('^Q',self.data_freq):

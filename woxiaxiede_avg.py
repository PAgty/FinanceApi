from transform import transform
import pandas as pd
from utility import infer_freq
import numpy as np


class woxiaxiede_avg(transform):
    diff = {}
    df = {}
    converted_df = {}

    def __init__(self):
        pass

    def fit(self, data, freq=None):
        self.time = data.keys()
        self.df = pd.Series(data)
        self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
        self.df = pd.to_numeric(self.df, errors='coerce')
        self.df = self.df.astype(float)
        if not freq:
            freq = infer_freq(self.df)
        self.freq = freq
        self.weight = self.compute_weight(self.df.index)
        self.df.index.freq = self.freq

    def apply(self, targetStartDate="", targetEndDate="", targetFreq=""):
        upsampled = self.df.resample('1D')
        self.converted_df = upsampled.interpolate(method='linear')
        self.adjusted_df = self.adjust(self.converted_df, self.df)
        return self.resultInTime(self.adjusted_df, targetStartDate, targetEndDate, targetFreq)

    def resultInTime(self, df, start, end, targetFreq):
        selectDf = df[start:end]
        finalDf = selectDf
        if targetFreq == 'M':
            finalDf = selectDf.resample('1M').agg(np.sum)
        if targetFreq == 'W':
            finalDf = selectDf.resample('W').agg(np.sum)
        if targetFreq == 'QS':
            finalDf = selectDf.resample('3M').agg(np.sum)
        if targetFreq == 'D':
            finalDf = selectDf

        finalDf.index = finalDf.index.strftime("%Y-%m-%d")
        dataDict = finalDf.to_dict()
        return dataDict

    def adjust(self, df, old_df):
        index = 0
        old_index = 0
        for i in self.weight:
            stride = i/2
            if i % 2 == 0:
                adj_1 = df.iloc[int(index + stride)] - old_df.iloc[int(old_index)]
                adj_2 = df.iloc[int(index + stride - 1)] - old_df.iloc[int(old_index)]
                adj = (adj_1 + adj_2)/2
            else:
                adj = df.iloc[int(index + stride)] - old_df.iloc[int(old_index)]
            df.iloc[int(index):int(index + i)] = df.iloc[int(index)                                                :int(index + i)] - adj
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
        return diff

    def append_index(self, df):
        # index_length = len(df)
        if df.index.freq:
                df[df.index[-1] + 1] = np.nan

        return df

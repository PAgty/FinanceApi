from algorithms import algorithm
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
from pyramid.arima import auto_arima

# p,q,i predict_time


class arima(algorithm):

    # dufault parameters
    data = {}
    output = []
    START_P = 1
    MAX_P = 3
    START_Q = 1
    MAX_Q = 3
    D = 1
    MAX_D = 2
    default_params = {
        'seasonal': False,
        'error_action': 'ignore',
        'suppress_warnings': True,
        'stepwise': True}

    def __init__(self, params):
        if params:
            for k, v in params.items():
                setattr(self, k, int(v))

    def fit(self, data):
        # get all parameters prepared
        self.data = self.preprocess(data)
        values = [x for x in self.data.values()]
        user_params = self.set_parameters(values)
        self.params = {**user_params, **self.default_params}
        # do the fitting

        # stepwise_model = auto_arima(values, start_p=1, start_q=1,
    #            max_p=3, max_q=3, m=12,
    #            start_P=0, seasonal=False,
    #            d=1, D=1, trace=True,
    #            error_action='ignore',
    #            suppress_warnings=True,
    #            stepwise=True)
# 		print(self.params)
        stepwise_model = auto_arima(**self.params)
        # self.model = ARIMA(values, order=parameters)
        # self.model_fit = self.model.fit(disp=0)
        self.model_fit = stepwise_model.fit(values)
        order = self.model_fit.get_params().get('order')
        print('selected parameters P: {} I : {} Q: {}'.format(*order))
        return self.model_fit

    def predict(self, time):

        if time:
            self.predict_time = time
        self.output = self.model_fit.predict(n_periods=self.predict_time)
        # self.output = self.model_fit.predict(start = start_index,end = end_index)
        # print(self.output)
        return self.output

    def set_parameters(self, data):
        params = {}
        params['y'] = data
        try:
            params['start_p'] = self.P
            params['max_p'] = self.P
        except BaseException:
            params['start_p'] = self.START_P
            params['max_p'] = self.MAX_P

        try:
            params['d'] = self.I
        except BaseException:
            params['d'] = self.D
            params['max_d'] = self.MAX_D

        try:
            params['start_q'] = self.Q
            params['max_q'] = self.Q
        except BaseException:
            params['start_q'] = self.START_Q
            params['max_q'] = self.MAX_Q

        return params

    def get_summary(self):
        try:
            s = self.model_fit.summary()
            return s
        except Exception as e:
            print(e)

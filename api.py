'''api version 1.0 implemented with flask_restFul api'''
import json
import redis
import pickle 

from flask import Flask
from flask_restful import Resource, Api, abort
from flask_restful import reqparse
from collections import OrderedDict
from itertools import islice

from woxiaxiede_agg import woxiaxiede_agg
from woxiaxiede_avg import woxiaxiede_avg
from Prophet import Prophet
from arima import arima
from utility import generateIndex, post_Check, buildDf, infer_freq


ALGORITHM_LIST = ['ARIMA', 'PROPHET']

# pylint: disable-msg=C0103
# db = redis.StrictRedis(host='10.103.245.241', port=6379,
#                        db=0, decode_responses=True)

db = redis.StrictRedis(host='localhost', port=6379,
                       db=0, decode_responses=True)
print('INDEXING generation start')
INDEXING = generateIndex(db)
print('INDEXING generation finished')


def abort_if_tag_doesnt_exist(tag):
    '''if the query tags is not in database, return 404,could be changed later'''
    if tag not in INDEXING:
        abort(404, message="tag {} doesn't exist in the current database".format(tag))
    return True


def abort_if_uuid_doesnt_exist(uuid):
    '''if the uuid doest not exist, return 404'''
    abort(404, message="uuid {} doesn't exist in the current database".format(uuid))


def abort_if_dataframe_format_incorrect():
    '''only accept json or csv format of dataframe, the index should be in the format of 'yy-mm-dd',otherwise return 400'''
    abort(400, message="provided dataframe is not supported, current api only accept json or csv format of dataframe, the index should be in the format of 'yy-mm-dd'")

def abort_if_transform_data_incorrect(e):
    '''only accept json format'''
    abort(400,message="please provide correct time series data in type dict and json format {}".format(e))

def abort_if_transform_time_incorrect(e):
    '''target timeframe cannot exceed provided time frame'''
    abort(400,message='the target start and end time must be within the time range of given input timeseries ')

class Query(Resource):
    '''this is the /query api for querying data'''

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument(
            'tags', type=str, required=True, help='The query tags', action='append')

    def get(self):
        '''GET method for querying data by intersection of tags, return 404 if ANY tag is not in the database'''
        args = self.parser.parse_args()
        tags = args.tags
        uuidList = [INDEXING[tag]
                    for tag in tags if abort_if_tag_doesnt_exist(tag)]
        uuidSet = set.intersection(*uuidList)
        return self.search(uuidSet)

    def search(self, uuidSet):
        '''function that return a list of all the selected dataframe'''
        dfList = [json.loads(db.get(i)) for i in uuidSet]
        return dfList


class Database(Resource):
    '''this is the /database api used to get the dataframe by their uuid'''

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('uuid', type=str, required=True,
                                 help='the index of time_series in database')
        self.keySet = set(db.keys())
        self.post_parser = reqparse.RequestParser()
        self.post_parser.add_argument(
            'dataframe', required=True, help='the time series you want to store, type should be json or cvs')
        self.post_parser.add_argument(
            'tags', required=True, help='list of tags of the time series', action='append')
        self.post_parser.add_argument('freq', type=str, required=True, choices=[
                                      'YS', 'D', 'MS', 'W', 'QS', 'H'], help='the frequency of the time series, options are [H, D, W, M, Q, Y], this field is not required')

    def get(self):
        '''the GET method used to get the dataframe'''
        args = self.parser.parse_args()
        i = args.uuid
        value = json.loads(
            db.get(i)) if i in self.keySet else abort_if_uuid_doesnt_exist(i)
        return value

    def post(self):
        '''the post method used to post new dataframe into the database'''
        args = self.post_parser.parse_args()
        dataframe = args.dataframe
        tags = args.tags
        freq = args.freq
        recognize, fileFormat, data = post_Check(dataframe)
        if recognize:
            dfList = buildDf(data, tags, freq, fileFormat)
            if dfList:
                for df in dfList:
                    db.set(df['uuid'], json.dumps(df))
                    dfTag = df['tags']
                    for t in dfTag:
                        if t not in INDEXING:
                            INDEXING[t] = []
                        INDEXING[t].append(df['uuid'])
                return dfList
            else:
                abort_if_dataframe_format_incorrect()
        else:
            abort_if_dataframe_format_incorrect()

class Backtest(Resource):
    '''this is the /test api used to backtest different algorithm'''
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument(
            'model', required=True,type = bytes, help='please provide the pickle dumpfile of the desired object'
        )
    def post(self):
        args = self.parser.parse_args()
        print(args.model)
        testingObject = pickle.loads(args.model)
        print(testingObject.__dir__())
        print(type(testingObject))
        return testingObject.__dir__()

class Indexing(Resource):
    '''this is the /indexing api used to get all the current available indexing'''

    def get(self):
        return list(set(INDEXING.keys()))

class Transform(Resource):
    '''this is the /transform api used to transform input time series'date frequent'''
    
    def __init__(self):
        self.targetStartDate = ''
        self.targetEndDate = ''
        self.parser = reqparse.RequestParser()
        self.parser.add_argument(
            'data',
            required = True,
            help = 'please provide the timeseries which will be transformed, in json format'
        )
        self.parser.add_argument(
            'originFrequent',
            help = 'the frequent of input data, optional'
        )
        self.parser.add_argument(
            'targetStartDate',
            type = str,
            help = "the desired start date of the new transformed data"
        )
        self.parser.add_argument(
            'targetEndDate',
            type = str,
            help = "the desired end date of the new transformted data"
        )
        self.parser.add_argument(
            'targetFrequent',
            required = True,
            type = str,
            choices = ['D','MS','QS','Y','W'],
            help = "please provide the target frequent of the new transformed data"
        )
        self.parser.add_argument(
            'type',
            required = True,
            type =str,
            choices = ['aggregate','average'],
            help = "please provide the data type from [aggregate,average]"
        )
    
    def post(self):
        args = self.parser.parse_args()
        args = self.transformArgsCheck(args)
        transformedData = self.data
        if self.type == 'aggregate':
            model = woxiaxiede_agg()
            model.fit(data,self.originFreq)
            transformedData = model.apply(
                targetStartDate = self.targetStartDate, targetEndDate = self.targetEndDate, targetFreq=self.targetFreq)
        if self.type == 'average':
            model = woxiaxiede_avg()
            model.fit(self.data, self.originFreq)
            transformedData = model.apply(
                targetStartDate=self.targetStartDate, targetEndDate=self.targetEndDate, targetFreq=self.targetFreq)
        return transformedData

    def transformArgsCheck(self,args):
        try:
            self.data = json.loads(args.data)
        except Exception as e:
            abort_if_transform_data_incorrect(e)
        try:
            self.od = OrderedDict(self.data)
        except Exception as e:
            abort_if_transform_data_incorrect(e)
        startDate = list(islice(self.od.items(), 0, 1))[0][0]
        endDate = list(islice(self.od.items(), len(self.od) - 1, len(self.od)))[0][0]
        self.targetStartDate = args.targetStartDate if args.targetStartDate else startDate
        self.targetEndDate = args.targetEndDate if args.targetEndDate else endDate
        if self.targetStartDate < startDate or self.targetEndDate > endDate:
            abort_if_transform_time_incorrect()
        self.type = args.type
        self.targetFreq = args.targetFrequent
        self.originFreq = args.originFrequent if args.originFrequent else infer_freq(self.data)

class Apply(Resource):
    '''this is the /apply api used to apply method on data'''

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument(
            'algorithm',
            type=str,
            required=True,
            help='the algorithm which will be applied to the data,currently:{}'.format(ALGORITHM_LIST))
        self.parser.add_argument(
            'params',
            type=dict,
            help="all the parameters used to the prediction algorithm")
        self.parser.add_argument(
            'data',
            required=True,
            type=str,
            help="data must be provided")
        self.parser.add_argument(
            'operator',
            type=str,
            required=True,
            help="the desired operation")
        self.parser.add_argument(
            'data_type',
            type=str,
            help="only used in tranform oepration,aggregate or average data")
        self.parser.add_argument('time', type=int, help="the prediction time")
        self.parser.add_argument(
            'weight',
            type=dict,
            help='user specified transformation weight')

    def post(self):
        '''apply the user specified operation on the data'''
        args = self.parser.parse_args()
        print(args.data)
        data = json.loads(args.data)
        operator = args.operator
        if operator == 'predict':
            return self.predict(algorithm=args.algorithm,
                                params=args.params, data=data, time=args.time)
        if operator == 'transform':
            return self.transform(data=args.data,
                                  data_type=args.data_type, time=args.time, weight=args.weight)

    def transform(self, data, data_type, time, weight):
        if data_type == 'aggregate':
            return self.aggregate(data, time, weight)
        if data_type == 'average':
            return self.average(data, time)

    def aggregate(self, data, time, weight):
        # if algorithm == "woxiaxiede_agg":
        agg_model = woxiaxiede_agg()
        agg_model.fit(data)
        result = agg_model.apply(time, weight)
        return result

    def average(self, data, time):
        # if algorithm == 'woxiaxiede_avg':
        avg_model = woxiaxiede_avg()
        avg_model.fit(data)
        result = avg_model.apply(time)
        return result

    def predict(self, algorithm, params, data, time):
        alg_name = algorithm.lower()
        if alg_name == 'arima':
            arima_model = arima(params)
            arima_model.fit(data)
            result = arima_model.predict(time)
            return result
        if alg_name == 'prophet':
            prophet_model = Prophet(params)
            prophet_model.fit(data)
            result = prophet_model.predict(time)
            print(result)
            return result


app = Flask(__name__)
api = Api(app)

# routing
api.add_resource(Query, '/query')
api.add_resource(Database, '/database')
api.add_resource(Apply, '/apply')
api.add_resource(Backtest,'/backtest')
api.add_resource(Indexing,'/index')
api.add_resource(Transform,'/transform')

if __name__ == '__main__':
    app.run(host = '0.0.0.0',debug=True)

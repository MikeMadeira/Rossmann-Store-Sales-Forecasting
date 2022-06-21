import pickle
import pandas as pd
import numpy as np
import math
import datetime
import re

class Rossmann( object ):
    
    def __init__(self):
        self.home_path = '/home/data-madness/Documents/SynologyDrive/ComunidadeDS/DSemProducao/Rossmann-Store-Sales-Forecasting/'
        self.competition_distance_scaler = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_encoder = pickle.load(open(self.home_path + 'parameter/store_type_encoder.pkl', 'rb'))
        
    def data_cleaning(self, df):
        
        ## Rename columns 
        
        columns = list(df.columns)
        
        cols_single_words = [re.findall("[A-Z][^A-Z]*", column) for column in columns]
        cols_lowercase_words = [list(map(lambda x: x.lower(),column)) for column in cols_single_words]
        cols_snakecase = ['_'.join(column) for column in cols_lowercase_words]
        
        df.columns = cols_snakecase
        
        ## Change date type
        
        df['date'] = pd.to_datetime(df['date'])
        
        ## Fill missing values
        
        # competition_distance
        orig_max_competition_dist = df.competition_distance.max()
        df.loc[df.competition_distance.isna(),'competition_distance'] = 200000.00

        # competition_open_since_month
        df.competition_open_since_month = df[['date','competition_open_since_month']].apply(lambda x: x[0].month if (math.isnan(x[1]))
                                                                                                                 else x[1],axis=1)
        # competition_open_since_year
        df.competition_open_since_year = df[['date','competition_open_since_year']].apply(lambda x: x[0].year if (math.isnan(x[1]))
                                                                                                              else x[1],axis=1)

        # promo2_since_week
        df.promo2_since_week = df[['date','promo2_since_week']].apply(lambda x: x[0].week if (math.isnan(x[1]))
                                                                                          else x[1],axis=1)

        # promo2_since_year 
        df.promo2_since_year = df[['date','promo2_since_year']].apply(lambda x: x[0].year if (math.isnan(x[1]))
                                                                                          else x[1],axis=1)
        # promo_interval
        month_dict = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        df.loc[df.promo_interval.isna(),'promo_interval'] = df.date.dt.month.map( month_dict )
    
        ## Change filled data type
        
        # competition_open_since_month to int
        df['competition_open_since_month'] = df['competition_open_since_month'].astype('int64')

        # competition_open_since_year to int
        df['competition_open_since_year'] = df['competition_open_since_year'].astype('int64')

        # promo2_since_week to int
        df['promo2_since_week'] = df['promo2_since_week'].astype('int64')

        # promo2_since_year to int
        df['promo2_since_year'] = df['promo2_since_year'].astype('int64')
        
        return df
    
    def feature_engineering(self, df):
        # year
        df['year'] = df['date'].dt.year

        # month
        df['month'] = df['date'].dt.month

        # day
        df['day'] = df['date'].dt.day

        # week_of_year
        df['week_of_year'] = df['date'].dt.week

        # year-week format
        df['year_week'] = df['date'].dt.strftime('%Y-%W')

        # competition_since
        df['competition_since'] = df.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],month=x['competition_open_since_month'],day=1), axis=1)
        df['competition_time_month'] = ( (df['date'] - df['competition_since'])/30 ).apply(lambda x: x.days).astype(int) 

        # promo_since
        # since the attributes promo2_since_year and week are integers and we want the as datetime format, we need to convert them into string first
        df['promo_since'] = df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week'].astype(str)
        # this will return for each date with Year-WeekofYear-WeekDay format as a string, the respective date in datetime Year-Month-Day format
        df['promo_since'] = df['promo_since'].apply(lambda x: datetime.datetime.strptime(x+'-1','%Y-%W-%w'))
        # calculate in how much weeks the competition has opened on each daily sales record 
        df['promo_time_week'] = ((df['date'] - df['promo_since'])/7).apply(lambda x: x.days ).astype(int)

        # assortment
        df['assortment'] = df['assortment'].apply(lambda x: 'basic' if x=='a'
                                                                    else 'extra' if x=='b'
                                                                    else 'extended') 

        # state_holiday
        df['state_holiday'] = df['state_holiday'].apply(lambda x: 'public_holiday' if x=='a'
                                                                                   else 'easter_holiday' if x=='b'
                                                                                   else 'christmas' if x=='c'
                                                                                   else 'regular_day')
        
        ## Filter unwanted records and features
        df = df[(df['open'] !=0)]
        
        # Drop the auxiliar columns from the feature engineering step and open since now all the records are from when the stores are opened
        cols_drop = ['open','competition_since','promo_since']

        df = df.drop(cols_drop,axis=1)
        
        return df
    
    def data_preparation(self, df):
        
        ## Rescaling
        # competition distance
        df['competition_distance'] = self.competition_distance_scaler.fit_transform(df[['competition_distance']].values)

        # competition time month
        df['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df[['competition_time_month']].values)

        # promo time week
        df['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df[['promo_time_week']].values)

        # year
        df['year'] = self.year_scaler.fit_transform(df[['year']].values)
        
        ## Feature encoding
        # state_holiday - One Hot Encoding
        df = pd.get_dummies(df, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df['store_type'] = self.store_type_encoder.fit_transform(df['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df['assortment'] = df['assortment'].map(assortment_dict)
        
        ## Cyclical features transformation
        # day_of_week
        n_day_of_week = len(df['day_of_week'].unique()) 
        df['day_of_week_sin'] = df['day_of_week'].apply(lambda x: np.sin(x*(2.*np.pi/n_day_of_week)))
        df['day_of_week_cos'] = df['day_of_week'].apply(lambda x: np.cos(x*(2.*np.pi/n_day_of_week)))

        # month
        n_month = len(df['month'].unique())
        df['month_sin'] = df['month'].apply(lambda x: np.sin(x*(2.*np.pi/n_month)))
        df['month_cos'] = df['month'].apply(lambda x: np.cos(x*(2.*np.pi/n_month)))

        # day
        n_day = len(df['day'].unique())
        df['day_sin'] = df['day'].apply(lambda x: np.sin(x*(2.*np.pi/n_day)))
        df['day_cos'] = df['day'].apply(lambda x: np.cos(x*(2.*np.pi/n_day)))

        # week_of_year
        n_weeks = len(df['week_of_year'].unique())
        df['week_of_year_sin'] = df['week_of_year'].apply(lambda x: np.sin(x*(2.*np.pi/n_weeks)))
        df['week_of_year_cos'] = df['week_of_year'].apply(lambda x: np.cos(x*(2.*np.pi/n_weeks)))
        
        ## Feature selection
        cols_selected = ['store','promo','store_type','assortment','competition_distance',
                         'promo2','competition_time_month','promo_time_week','day_of_week_sin','day_of_week_cos',
                         'month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos']
        
        return df[cols_selected]
    
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        
        original_data_df = original_data.copy(deep=True)

        # also filter original data unwanted records   
        original_data_df = original_data_df[(original_data_df['open'] !=0)]

        #print(pred)
        # join pred into the original data | review the log and exp transformation
        original_data_df['prediction'] = np.exp(pred)
        
        return original_data_df.to_json(orient='records', date_format='iso')

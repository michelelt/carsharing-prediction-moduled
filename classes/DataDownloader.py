#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:57:58 2019

@author: mc
"""

import ssl
import pymongo
import pandas as pd
import datetime
from math import *
import time

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from GlobalsFunctions import haversine, crs_

import matplotlib.pyplot as plt
 


class DataDownloader:
    
    def __init__(self, city, provider):
        self.city = city
        self.provider = provider
        self.parking_collection = 'PermanentParkings'
        self.booking_collection = 'PermanentBookings'
        
        if self.provider.lower() ==  'enjoy':
            self.parking_collection = 'enjoy_'+self.parking_collection
            self.booking_collection = 'enjoy_'+self.booking_collection

        credential = open('../../credential/credential.txt', 'r')
        lines = credential.readlines()
        credential.close()

        self.connection_data = {}
        for line in lines:
            split = line.split(':')
            self.connection_data[split[0]] = split[1][:-1]

            
    def setup_mongodb(self, collection):
        try:
            client = pymongo.MongoClient(self.connection_data['server'],
                                         int(self.connection_data['port']),
                                         ssl=bool(self.connection_data['ssl']),
                                         ssl_cert_reqs=ssl.CERT_NONE) # server.local_bind_port is assigned local port                #client = pymongo.MongoClient()
            client.server_info()
            db = client[self.connection_data['db']] #Choose the DB to use
            db.authenticate(self.connection_data['user'], self.connection_data['password'])#, mechanism='MONGODB-CR') #authentication         #car2go_debug_info = db['DebugInfo'] #Collection for Car2Go watch
            Collection = db[collection] #Collection for Enjoy watch
        except pymongo.errors.ServerSelectionTimeoutError as err:
            print(err)
    
        return Collection
    
    
    def post_process_data(self, bookings_df):
    	print("Post processing data beginning")
    	bookings_df["duration"] = bookings_df["final_time"] - bookings_df["init_time"]
    	bookings_df["duration"] = bookings_df["duration"].astype(int)
    	bookings_df = bookings_df.drop('driving',1)
    	print('Duration computed')
    
    
    	bookings_df['type'] = bookings_df.origin_destination.apply(lambda x : x['type'])
    	bookings_df['coordinates'] = bookings_df.origin_destination.apply(lambda x : x['coordinates'])
    	bookings_df = bookings_df.drop('origin_destination',1)
    	print('Type and coordinates separated')
    
    
    	bookings_df['end'] = bookings_df.coordinates.apply(lambda x : x[0])
    	bookings_df['start'] = bookings_df.coordinates.apply(lambda x : x[1])
    	bookings_df = bookings_df.drop('coordinates',1)
    	print('Separated start and end')
    
    
    	bookings_df['start_lon'] = bookings_df.start.apply(lambda x : float(x[0]) )
    	bookings_df['start_lat'] = bookings_df.start.apply(lambda x : float(x[1]) )
    	bookings_df = bookings_df.drop('start',1)
    	print('Start lat and start lon separated')
    
    	bookings_df['end_lon'] = bookings_df.end.apply(lambda x : float(x[0]) )
    	bookings_df['end_lat'] = bookings_df.end.apply(lambda x : float(x[1]) )
    	bookings_df = bookings_df.drop('end', 1)
    	print('End lat and end lon separated')
    
    
    	bookings_df['distance'] = bookings_df.apply(lambda x : haversine(
    	       float(x['start_lon']),float(x['start_lat']),
    	       float(x['end_lon']), float(x['end_lat'])), axis=1)
    	print('Distance computed')
    
    	bookings_df['Year'] = pd.DatetimeIndex(bookings_df['init_date']).year
    	bookings_df['Month'] = pd.DatetimeIndex(bookings_df['init_date']).month
    	bookings_df['Day'] = pd.DatetimeIndex(bookings_df['init_date']).day
    	bookings_df['Hour'] = pd.DatetimeIndex(bookings_df['init_date']).hour
    
    	print('DONE!')
    	return bookings_df  
    
    
    def query_data(self, collection, *args, **kwargs):
        i_date = kwargs.get('i_date', None)
        f_date = kwargs.get('f_date', None)
        
        if (not isinstance(i_date, datetime.datetime)
            and not i_date is None):
            print('Wrong init date format')
            return
        
        if (not isinstance(f_date, datetime.datetime)
            and not f_date is None):
            print('Wrong final date format')
            return
        if i_date is None: i_date = datetime.datetime(1991, 8, 20, 11, 40, 0)
        if f_date is None: f_date = datetime.datetime(2191, 8, 20, 11, 40, 0)
        
        collection = self.setup_mongodb(collection)
        
        print('Queried city: %s'%self.city)
        print('Query time interval:')
        print('Init date: ', i_date)
        print('Final date:', f_date)
        
        
        output = collection.find({"city": self.city, 
                                     "init_date": {"$gt": i_date, 
                                                   "$lt": f_date }
                                   })

        bookings = pd.DataFrame(list(output))
        
        bookings = self.post_process_data(bookings)
        
        return bookings
    



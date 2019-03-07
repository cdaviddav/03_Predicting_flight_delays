
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:40:39 2017

@author: cdavid
"""

""" Import all packages and the used settings and functions """

import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


from settings import Settings
from src.functions.data_preprocessing import *
from src.functions.functions_plot import *
from src.functions.regression_pipeline import *

settings = Settings()


""" ---------------------------------------------------------------------------------------------------------------
Load training and test dataset
"""

# Load train datasets
airlines = pd.read_csv(settings.config['Data Locations'].get('airlines'))
airport = pd.read_csv(settings.config['Data Locations'].get('airport'))
flights = pd.read_csv(settings.config['Data Locations'].get('flights'))

airlines.name = 'airlines'
airport.name = 'airport'
flights.name = 'flights'




""" ---------------------------------------------------------------------------------------------------------------
Explore the data
First take a look at the training dataset
- what are the features and how many features does the training data include
- are the missing values (but take a deeper look at the data preperation process)
- what are the different units of the features
"""

# Get a report of the training and test dataset as csv
# -> Use the function describe_report(df, name, output_file_path=None)
describe_report(airlines, output_file_path=settings.csv)
describe_report(airport, output_file_path=settings.csv)
describe_report(flights, output_file_path=settings.csv)


airlines, airport, flights =  drop_missing_values(airlines, airport, flights, None, output_file_path = settings.csv)


def detailed_analysis(flights, airlines, output_file_path=None):
    # -------- Detailed analysis: MONTH --------
    flights_MONTH = flights['MONTH'].value_counts()
    flights_MONTH = flights_MONTH.sort_values()

    fig1 = plt.figure(figsize=(16, 8))
    ax1 = fig1.add_subplot(111)
    flights_MONTH.plot(kind='bar', ax=ax1, title="Verteilung Anzahl der Flüge über Monate")
    ax1.set_frame_on(True)

    # Customize title, set position, allow space on top of plot for title
    ax1.set_title(ax1.get_title(), fontsize=25, alpha=0.7, ha='left')
    plt.subplots_adjust(top=0.9)
    ax1.title.set_position((0, 1.01))

    # Set y axis label on top of plot, set label text
    ax1.yaxis.set_label_position('left')
    ylab = 'Anzahl Flüge'
    ax1.set_ylabel(ylab, fontsize=20, alpha=0.7, ha='left')
    ax1.yaxis.set_label_coords(-0.07, 0.5)

    ax1.xaxis.set_tick_params(which='major', labelsize=15, rotation=0)
    ax1.yaxis.set_tick_params(which='major', labelsize=15)

    fig1.savefig(os.path.join(output_file_path, 'flights_MONTH.pdf'), bbox_inches='tight', dpi=800)


    # -------- Detailed analysis: DAY_OF_WEEK --------
    DAY_OF_WEEK_Map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday',
                       7: 'Sunday'}
    flights_DAY_OF_WEEK = flights['DAY_OF_WEEK'].value_counts()
    flights_DAY_OF_WEEK = flights_DAY_OF_WEEK.rename(index=DAY_OF_WEEK_Map)

    fig2 = plt.figure(figsize=(16, 8))
    ax2 = fig2.add_subplot(111)
    flights_DAY_OF_WEEK.plot(kind='bar', ax=ax2, title="Verteilung Anzahl der Flüge über Monate", sort_columns=True)
    ax2.set_frame_on(True)

    # Customize title, set position, allow space on top of plot for title
    ax2.set_title(ax2.get_title(), fontsize=25, alpha=0.7, ha='left')
    plt.subplots_adjust(top=0.9)
    ax2.title.set_position((0, 1.01))

    # Set y axis label on top of plot, set label text
    ax2.yaxis.set_label_position('left')
    ylab = 'Anzahl Flüge'
    ax2.set_ylabel(ylab, fontsize=20, alpha=0.7, ha='left')
    ax2.yaxis.set_label_coords(-0.07, 0.5)

    ax2.xaxis.set_tick_params(which='major', labelsize=15, rotation=70)
    ax2.yaxis.set_tick_params(which='major', labelsize=15)

    fig2.savefig(os.path.join(output_file_path, 'flights_DAY_OF_WEEK.pdf'), bbox_inches='tight', dpi=800)


    # -------- Detailed analysis: AIRLINE --------
    AIRLINE_Map = airlines.to_dict()
    flights_AIRLINE = flights['AIRLINE'].value_counts()
    flights_AIRLINE = flights_AIRLINE.rename(index=AIRLINE_Map['AIRLINE'])

    fig3 = plt.figure(figsize=(16, 8))
    ax3 = fig3.add_subplot(111)
    flights_AIRLINE.plot(kind='bar', ax=ax3, title="Airlines", sort_columns=True)
    ax3.set_frame_on(True)

    # Customize title, set position, allow space on top of plot for title
    ax3.set_title(ax3.get_title(), fontsize=25, alpha=0.7, ha='left')
    plt.subplots_adjust(top=0.9)
    ax3.title.set_position((0, 1.01))

    # Set y axis label on top of plot, set label text
    ax3.yaxis.set_label_position('left')
    ylab = 'Anzahl Flüge'
    ax3.set_ylabel(ylab, fontsize=20, alpha=0.7, ha='left')
    ax3.yaxis.set_label_coords(-0.07, 0.5)

    ax3.xaxis.set_tick_params(which='major', labelsize=15, rotation=70)
    ax3.yaxis.set_tick_params(which='major', labelsize=15)

    fig3.savefig(os.path.join(output_file_path, 'flights_AIRLINE.pdf'), bbox_inches='tight', dpi=800)


    # -------- Detailed analysis: ORIGIN_AIRPORT --------
    flights_ORIGIN_AIRPORT = flights['ORIGIN_AIRPORT'].value_counts()
    flights_ORIGIN_AIRPORT = flights_ORIGIN_AIRPORT.nlargest(30)

    fig4 = plt.figure(figsize=(16, 8))
    ax4 = fig4.add_subplot(111)
    ax4 = flights_ORIGIN_AIRPORT.plot(kind='bar', ax=ax4, title="Origin Airport", sort_columns=True)
    ax4.set_frame_on(True)

    # Customize title, set position, allow space on top of plot for title
    ax4.set_title(ax4.get_title(), fontsize=25, alpha=0.7, ha='left')
    plt.subplots_adjust(top=0.9)
    ax4.title.set_position((0, 1.01))

    # Set y axis label on top of plot, set label text
    ax4.yaxis.set_label_position('left')
    ylab = 'Anzahl Flüge'
    ax4.set_ylabel(ylab, fontsize=20, alpha=0.7, ha='left')
    ax4.yaxis.set_label_coords(-0.07, 0.5)

    ax4.xaxis.set_tick_params(which='major', labelsize=15, rotation=70)
    ax4.yaxis.set_tick_params(which='major', labelsize=15)

    fig4.savefig(os.path.join(output_file_path, 'flights_ORIGIN_AIRPORT.pdf'), bbox_inches='tight', dpi=800)


#detailed_analysis(flights, airlines,  output_file_path = settings.figures)



""" ---------------------------------------------------------------------------------------------------------------
Feature Creation
"""
# Merge airport data to flight data
airport_ORIGIN = airport.copy()
airport_ORIGIN.columns = [str(col) + '_ORIGIN' for col in airport_ORIGIN.columns]
airport_ORIGIN = airport_ORIGIN.reset_index()
airport_ORIGIN = airport_ORIGIN.rename(columns={'IATA_CODE_ORIGIN': 'ORIGIN_AIRPORT'})

airport_DESTINATION = airport.copy()
airport_DESTINATION.columns = [str(col) + '_DESTINATION' for col in airport_DESTINATION.columns]
airport_DESTINATION = airport_DESTINATION.reset_index()
airport_DESTINATION = airport_DESTINATION.rename(columns={'IATA_CODE_DESTINATION': 'DESTINATION_AIRPORT'})

flights = pd.merge(flights, airport_ORIGIN, on='ORIGIN_AIRPORT')
flights = pd.merge(flights, airport_DESTINATION, on='DESTINATION_AIRPORT')

del airport_ORIGIN, airport_DESTINATION, airport



# Calculate haversine distance
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h # in km

flights['distance_haversine'] = haversine_array(
        flights['LATITUDE_ORIGIN'].values, flights['LONGITUDE_ORIGIN'].values,
        flights['LATITUDE_DESTINATION'].values, flights['LONGITUDE_DESTINATION'].values)



""" ---------------------------------------------------------------------------------------------------------------
Feature Selection
"""
from sklearn.utils import shuffle
flights, _, _ = drop_missing_values(flights, None, None, None, output_file_path = settings.csv)
flights = shuffle(flights)
flights = flights[:50000]

flights = labelEnc(flights)

# split dataset to features and lables
y_train = flights['ARRIVAL_DELAY']
x_train = flights.drop(['ARRIVAL_DELAY', 'TAIL_NUMBER', 'AIRPORT_ORIGIN', 'AIRPORT_DESTINATION'], axis=1)



""" ---------------------------------------------------------------------------------------------------------------
Machine Learning (Regression)
"""

# split in training and testing datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)


pipeline = Pipeline([
    ('reduce_dim', PCA()),
    ('feature_scaling', MinMaxScaler()), # scaling because linear models are sensitive to the scale of input features
    ('regression', Ridge()), # Ridge is default algorithm for linear models
    ])

param_grid = [{'reduce_dim__n_components': [10, 14],
               'regression__alpha': [0.005, 0.05, 0.1, 0.5],
               'regression__solver': ['svd', 'cholesky']
              }]


pipe_best_params = regression_pipeline(x_train, y_train, pipeline, 5, 'r2', param_grid)

pipe_best = Pipeline([
    ('reduce_dim', PCA(n_components = pipe_best_params['reduce_dim__n_components'])),
    ('feature_scaling', MinMaxScaler()),
    ('regression', Ridge(alpha = pipe_best_params['regression__alpha'],solver = pipe_best_params['regression__solver'])),
])

print(pipe_best_params['reduce_dim__n_components'])
print(pipe_best_params['regression__alpha'])
print(pipe_best_params['regression__solver'])

train_errors = evaluate_pipe_best_train(x_train, y_train, pipe_best, 'Ridge', log=False, output_file_path=settings.figures)

plot_learning_curve(pipe_best, 'Ridge', x_train, y_train, 'r2', output_file_path=settings.figures)



""" ---------------------------------------------------------------------------------------------------------------
Evaluate the System on the Test Set
"""
#Evaluate the model with the test_set
# -> Use the function evaluate_pipe_best_test(x_train, y_train, pipe_best, algo, output_file_path=None)
evaluate_pipe_best_test(x_test, y_test, pipe_best, 'Ridge', log=False, output_file_path=settings.figures)

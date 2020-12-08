from typing import Optional, List
from pathlib import Path
from datetime import datetime
import pytz
import json
import os

import pandas as pd
import numpy as np
from . import state2abbr, abbr2state



def load_cdc_truth(
    death: bool = False,
    cumulative: bool = True,
    start_date: str = '2020-01-23',
    end_date: Optional[str] = None,
):
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series" 
    path = f"{url}/time_series_covid19_{'deaths' if death else 'confirmed'}_US.csv"
    
    df = pd.read_csv(path)
    data = {}
    for state in state2abbr:
        tmp = df[df['Province_State']==state].loc[:, df.columns[(12 if death else 11):]].sum(axis=0)
        tmp.index = pd.to_datetime(tmp.index)
        data[state] = tmp
    data = pd.DataFrame(data)
    if not cumulative:
        data = data.diff(1).iloc[1:]
    if end_date is not None:
        end_date = pd.to_datetime(end_date) - pd.Timedelta(1, unit='d')
    data = data.loc[start_date:end_date]
    return data

def load_hosps_truth(
    start_date: str = '2020-07-01',
    end_date: Optional[str] = None,
):
    if end_date is None:
        end_date = pd.to_datetime(f"{datetime.now(pytz.timezone('US/Pacific')):%Y-%m-%d}")
    else:
        end_date = pd.to_datetime(end_date)
    file_date = f"{end_date - pd.Timedelta((end_date.weekday()-6)%7, unit='d'):%Y%m%d}"
    end_date = f"{end_date-pd.Timedelta(1, unit='d'):%Y-%m-%d}"
    url = f"https://healthdata.gov/sites/default/files/reported_hospital_utilization_timeseries_{file_date}_2146.csv"
    df = pd.read_csv(
        url, 
        parse_dates=['date'],
        usecols=[
            'state','date',
            'previous_day_admission_adult_covid_confirmed',
            'previous_day_admission_pediatric_covid_confirmed',
        ],
        index_col='date',
    ).sort_index()
    data = pd.DataFrame({
        state: data['previous_day_admission_adult_covid_confirmed']+data['previous_day_admission_pediatric_covid_confirmed']
        for state, data in df.groupby('state')
    })
    data = data.loc[start_date:end_date, abbr2state.keys()].fillna(0.0)
    return data

    
def load_cases_baselines(
    date: str,
    est: str = 'point',
):
    time_field = 'target_end_date'
    date = f"{pd.to_datetime(date):%Y-%m-%d}"
    if pd.to_datetime(date) <= pd.to_datetime('2020-09-21'):
        url = f"https://www.cdc.gov/coronavirus/2019-ncov/covid-data/files/{date}-all-forecasted-cases-model-data.csv"
    else:
        url = f"https://www.cdc.gov/coronavirus/2019-ncov/downloads/cases-updates/{date}-all-forecasted-cases-model-data.csv"
    baselines = pd.read_csv(
        url,
        parse_dates=[time_field],
    )
    cdc = {}
    for model, data in baselines.groupby('model'):
        data = data.loc[data.fips.str.isnumeric()]
        # filter national-only
        if data.shape[0] == 0:
            continue
        if min(data.fips.str.len()) > 2:
            # aggreate county-only
            data = data.groupby(['State', time_field]).sum().reset_index()
            data['location_name'] = data['State'].apply(lambda x: abbr2state.get(x, x))
        else:
            # take state-level
            data = data[data.fips.str.len() <= 2]
        dfs = []
        for state, df in data.groupby('location_name'):
            df = df.loc[:, ['target_end_date', est]].set_index('target_end_date')
            df.index.name = 'date'
            df.columns = [state]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        cdc[model] = data
    return cdc

def load_hosps_baselines(
    date: str,
    est: str = 'point',
):
    time_field = 'target_end_date'
    date = f"{pd.to_datetime(date):%Y-%m-%d}"
    baselines = pd.read_csv(
        f'https://www.cdc.gov/coronavirus/2019-ncov/downloads/cases-updates/{date}-hospitalizations-model-data.csv',
        parse_dates = [time_field],
    )
    cdc = {}
    for model, data in baselines.groupby('model'):
        dfs = []
        for state, df in data.groupby('location_name'):
            df = df.loc[:, [time_field, est]].set_index(time_field)
            df.index.name = 'date'
            df.columns = [state]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        cdc[model] = data
    return cdc

    
def load_death_baselines(
    date: str,
    est: str = 'point',
):
    time_field = 'target_week_end_date'
    date = f"{pd.to_datetime(date):%Y-%m-%d}"
    baselines = pd.read_csv(
        f'https://www.cdc.gov/coronavirus/2019-ncov/covid-data/files/{date}-model-data.csv',
        parse_dates = [time_field],
    )
    baselines = baselines[baselines['target'].isin([f'{i} wk ahead inc death' for i in range(1,5)])]
    cdc = {}
    for model, data in baselines.groupby('model'):
        dfs = []
        for state, df in data.groupby('location_name'):
            df = df.loc[:, [time_field, est]].set_index(time_field)
            df.index.name = 'date'
            df.columns = [state]
            dfs.append(df)
        data = pd.concat(dfs, axis=1)
        cdc[model] = data
    return cdc

    
def load_cal_cases_data(
    death: bool,
    cumulative: bool,
    start_date: str = '2020-03-18',
    end_date: Optional[str] = None,
):
    url = 'https://data.ca.gov/dataset/590188d5-8545-4c93-a9a0-e230f0db7290/resource/926fd08f-cc91-4828-af38-bd45de97f8c3/download/statewide_cases.csv'
    df = pd.read_csv(url, parse_dates=['date'])
    col = f"{'total' if cumulative else 'new'}count{'deaths' if death else 'confirmed'}"
    data = {}
    for county, frame in df.groupby('county'):
        data[county] = frame.set_index('date').loc[:, col]
    data = pd.DataFrame(data)
    data.fillna(0, inplace=True)
    if end_date is not None:
        end_date = pd.to_datetime(end_date) - pd.Timedelta(1, unit='d')
    return data[start_date:end_date]


def load_cal_hosps_data(
    confirmed: Optional[bool] = None,
    icu: bool = False,
    start_date: str = '2020-03-29',
    end_date: Optional[str] = None,
):
    url = 'https://data.ca.gov/dataset/529ac907-6ba1-4cb7-9aae-8966fc96aeef/resource/42d33765-20fd-44b8-a978-b083b7542225/download/hospitals_by_county.csv'
    df = pd.read_csv(url, parse_dates=['todays_date'])
    if confirmed is None:
        col = [
            f"{'icu' if icu else 'hospitalized'}_covid_confirmed_patients",
            f"{'icu' if icu else 'hospitalized'}_suspected_covid_patients",
        ]
    else:
        col = f"{'icu' if icu else 'hospitalized'}_{'covid_confirmed' if confirmed else 'suspected_covid'}_patients"
    data = {}
    for county, frame in df.groupby('county'):
        frame = frame.set_index('todays_date')
        frame.fillna(0.0, inplace=True)
        if confirmed is None:
            frame = frame.loc[:, col].sum(axis=1)
        else:
            frame = frame.loc[:, col]
        data[county] = frame
    data = pd.DataFrame(data)
    if end_date is not None:
        end_date = pd.to_datetime(end_date) - pd.Timedelta(1, unit='d')
    return data.loc[start_date:end_date]
import pandas as pd
import json

with open('model/metadata.json', 'r') as file:
   metadata = json.load(file)


#STEP 1 extract hits features
def hits_features (df_hits):

    df_hits['is_car_page'] = df_hits['hit_page_path'].str.contains(r'sberauto.com/cars/all/[^/]+/[^/]+', na=False)

    # returning feature
    checked_car = df_hits.groupby('session_id')['is_car_page'].max().astype(int).reset_index()

    car_hits = df_hits[df_hits['is_car_page']].copy()

    car_hits[['brand', 'model']] = car_hits['hit_page_path'].str.extract(r'sberauto.com/cars/all/([^/]+)/([^/]+)')

    car_hits_sorted = car_hits.sort_values(['session_id', 'hit_number'])

    # returning feature
    last_car_by_session = car_hits_sorted.groupby('session_id').tail(1)[['session_id',
                                                                         'brand',
                                                                         'model']].reset_index()

    car_hits['full_model'] = car_hits['brand'] + ' ' + car_hits['model']

    # returning feature
    unique_cars_per_session = (
        car_hits.groupby('session_id')['full_model']
        .nunique()
        .reset_index(name='n_unique_cars')
    )

    hits_range = df_hits.groupby('session_id')['hit_number'].agg(['min', 'max'])
    # returning feature
    hits_range['hit_range'] = hits_range['max'] - hits_range['min']

    left_features = checked_car \
        .merge(hits_range['hit_range'], on = 'session_id', how = 'left') \
        .merge(last_car_by_session, on = 'session_id', how = 'left') \
        .merge(unique_cars_per_session, on = 'session_id', how = 'left')

    return left_features

#STEP 2 merge w df_sessions

#STEP 3 transform ad-features
def ad_features(df_sessions):

    ad_cols = ['utm_adcontent', 'utm_campaign', 'utm_source', 'utm_medium']

    #if df_sessions[ad_cols].isna():
    df_sessions[ad_cols] = df_sessions[ad_cols].fillna('(none)')
        #return df_sessions

    for col in ad_cols:
        top = metadata['top_15_ad_features'][col]
        df_sessions[col] = df_sessions[col].apply(
            lambda x: x if x in top else 'other')

    return df_sessions

#STEP 4 transform device features
def device_features (df_sessions):
    device_brand_top = metadata['device_brand_top']
    df_sessions['device_brand_new'] = df_sessions.device_brand.apply(lambda x: x if x in device_brand_top else '(not set)')

    def fill_os(row):
        if row['device_brand'] == 'Apple' and row['device_category'] != 'desktop':
            return 'iOS'
        elif row['device_brand'] == 'Apple' and row['device_category'] == 'desktop':
            return 'Macintosh'
        elif row['device_brand'] in ['Samsung',
                                     'Xiaomi',
                                     'Huawei',
                                     'Realme',
                                     'OPPO',
                                     'Vivo',
                                     'OnePlus',
                                     'Asus',
                                     'Nokia',
                                     'Sony',
                                     'ZTE',
                                     'Google',
                                     'Meizu'] and row['device_category'] != 'desktop':
            return 'Android'
        else:
            return '(not set)'

    df_sessions['device_os_new'] = df_sessions['device_os'].fillna(df_sessions.apply(fill_os, axis=1))

    def resolution_class(res_str):
        try:
            w, h = map(int, res_str.split('x'))
            pixels = w * h
            if pixels < 480 * 800:
                return 'low'
            elif pixels < 1080 * 1920:
                return 'medium'
            else:
                return 'high'
        except:
            return 'unknown'

    df_sessions['resolution_class'] = df_sessions['device_screen_resolution'].apply(resolution_class)

    df_sessions = df_sessions.drop(['device_brand', 'device_os', 'device_screen_resolution'], axis = 1)

    return df_sessions

#STEP 5 transform date features
def date_features(df_sessions):
    df_sessions['visit_hour'] = pd.to_datetime(df_sessions['visit_time'],
                                               format='%H:%M:%S', errors='coerce').dt.hour
    df_sessions['visit_date'] = pd.to_datetime(df_sessions['visit_date'], errors='coerce')

    df_sessions['visit_date_month'] = df_sessions.visit_date.dt.month
    df_sessions['visit_date_dayofweek'] = df_sessions.visit_date.dt.weekday
    df_sessions['visit_date_day'] = df_sessions.visit_date.dt.day

    df_sessions = df_sessions.drop(['visit_date', 'visit_time'], axis = 1)
    return df_sessions

#STEP 5 transform geo features
def geo_features(df_sessions):
    top_country = metadata['top_country']
    df_sessions['geo_country_new'] = df_sessions.geo_country.apply(lambda x: x if x in top_country else 'other')

    major = metadata['major_cities']
    df_sessions['geo_city_new'] = df_sessions['geo_city'].apply(lambda x: x if x in major else 'other')

    df_sessions = df_sessions.drop(['geo_country', 'geo_city'], axis=1)

    return df_sessions

#STEP 5 cut numerical anomalies
def del_outliers(df_sessions):
    visit_number_threshold = metadata['visit_number_threshold']
    hit_range_threshold = metadata['hit_range_threshold']

    df_sessions['visit_number'] = df_sessions['visit_number'].clip(upper = int(visit_number_threshold)+1)
    df_sessions['hit_range'] = df_sessions['hit_range'].clip(upper = int(hit_range_threshold)+1)

    return df_sessions

def prepare_data(df_hits, df_sessions):

    df_sessions = ad_features(df_sessions)
    df_sessions = device_features(df_sessions)
    df_sessions = date_features(df_sessions)
    df_sessions = geo_features(df_sessions)

    left_features = hits_features(df_hits)
    df_sessions= df_sessions.merge(left_features, on = 'session_id', how = 'left')

    df_sessions['n_unique_cars'] = df_sessions['n_unique_cars'].fillna(0)
    df_sessions[['brand', 'model']] = df_sessions[['brand', 'model']].fillna('(not set)')
    df_sessions = del_outliers(df_sessions)

    X = df_sessions.drop(['index'], axis = 1)
  #  print(X.shape, X.columns)
    return X

if __name__ == '__main__':
    prepare_data()
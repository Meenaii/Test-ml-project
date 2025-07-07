import pandas as pd
import json
#STEP 1 extract hits features
def hits_features (df_hits):

    success = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
               'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
               'sub_submit_success', 'sub_car_request_submit_click']

    df_hits['target'] = df_hits.event_action.apply(lambda x: int(x in success))

    # returning feature
    target_by_session = df_hits.groupby('session_id')['target'].max().astype(int).reset_index()

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

    return target_by_session, left_features

#STEP 2 do a dance

#STEP 3 transform ad-features
def ad_features(df_sessions):
    df_sessions = df_sessions.copy()
    ad_cols = ['utm_adcontent', 'utm_campaign', 'utm_source', 'utm_medium']
    df_sessions[ad_cols] = df_sessions[ad_cols].fillna('(none)')
    top={}
    for col in ad_cols:
        top[col] = df_sessions[col].value_counts().head(15).index.tolist()
        df_sessions[col] = df_sessions[col].apply(
            lambda x: x if x in top[col] else 'other')

    return df_sessions, top

#STEP 4 transform device features
def device_features (df_sessions):
    device_brand_top = df_sessions.device_brand.value_counts().head(15).index.tolist()
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

    return df_sessions, device_brand_top

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

#STEP 6 transform geo features
def geo_features(df_sessions):
    top_country = df_sessions.geo_country.value_counts().head(10).index.tolist()
    df_sessions['geo_country_new'] = df_sessions.geo_country.apply(lambda x: x if x in top_country else 'other')

    counts = df_sessions['geo_city'].value_counts(normalize=True)
    major_cities = counts[counts.cumsum() <= 0.9].index.tolist()
    df_sessions['geo_city_new'] = df_sessions['geo_city'].apply(lambda x: x if x in major_cities else 'other')

    df_sessions = df_sessions.drop(['geo_country', 'geo_city'], axis=1)

    return df_sessions, top_country, major_cities

#STEP 5 cut numerical anomalies
def del_outliers(df_sessions):
    visit_number_threshold =  float(df_sessions.visit_number.quantile(0.99))
    hit_range_threshold = float(df_sessions.hit_range.quantile(0.99))
    df_sessions = df_sessions[(df_sessions.visit_number < visit_number_threshold)]
    df_sessions = df_sessions[df_sessions.hit_range < hit_range_threshold]

    return df_sessions, visit_number_threshold, hit_range_threshold

def prepare_data():
    df_hits = pd.read_csv('data/ga_hits.csv').drop(['event_value',
                                                    'hit_referer',
                                                    'hit_time',
                                                    'event_label'], axis=1)

    df_sessions =  pd.read_csv('data/ga_sessions.csv', low_memory=False).drop(['device_model',
                                                                          'utm_keyword'], axis = 1)

    df_sessions, top_15_ad = ad_features(df_sessions)
    df_sessions, device_brand_top = device_features(df_sessions)
    df_sessions = date_features(df_sessions)
    df_sessions, top_country, major_cities = geo_features(df_sessions)

    target_by_session, left_features = hits_features(df_hits)
    df_sessions = df_sessions.merge(target_by_session, on = 'session_id', how = 'inner')
    df_sessions= df_sessions.merge(left_features, on = 'session_id', how = 'left')

    df_sessions['n_unique_cars'] = df_sessions['n_unique_cars'].fillna(0)
    df_sessions[['brand', 'model']] = df_sessions[['brand', 'model']].fillna('(not set)')

    df_sessions, visit_number_threshold, hit_range_threshold = del_outliers(df_sessions)

    y = df_sessions['target']
    X = df_sessions.drop(['target', 'session_id', 'client_id', 'index'], axis = 1)



    metadata = {'top_15_ad_features': top_15_ad,
                'device_brand_top': device_brand_top,
                'top_country': top_country,
                'major_cities': major_cities,
                'visit_number_threshold': visit_number_threshold,
                'hit_range_threshold': hit_range_threshold
                }

    with open('model/metadata.json', 'w', encoding='utf-8') as file:
        json.dump(metadata, file)
    #df_sessions.to_csv('data/prepared.csv', index = False)
    print('finished preprocess')

    return X, y

if __name__ == '__main__':
    prepare_data()
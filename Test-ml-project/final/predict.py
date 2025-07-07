import pandas as pd
import dill
from model.preprocess_predict import prepare_data

with open('model/classifier_pipe.pkl', 'rb') as file:
   model = dill.load(file)
#Загрузка данных
def main():
    df_hits = pd.read_csv('predict_data/ga_hits_session1.csv').drop(['event_value',
                                                    'hit_referer',
                                                    'hit_time',
                                                    'event_label'], axis=1)

    df_sessions =  pd.read_csv('predict_data/ga_sessinons_session1.csv', low_memory=False).drop(['device_model',
                                                                          'utm_keyword'], axis = 1)
    data = prepare_data(df_hits, df_sessions)
    sessions_info = data[['session_id', 'client_id']].copy()
    X = data.drop(['session_id', 'client_id'], axis=1)
    print('roc_auc_score:', model['metadata']['roc_auc'])
    y = model['model'].predict(X)
    proba = model['model'].predict_proba(X)
    #Вывод и сохранение данных
    if data.shape[0] == 1:
        print('Predicted_class:', y[0])
        print('Probability:', proba[0][y][0])
        sessions_info['Prediction'] = y
        sessions_info['Probability_of_1'] = proba[:,1]
        sessions_info.to_csv('predict_data/predictions.csv', index = False)
    else:
        sessions_info['Prediction'] = y
        sessions_info['Probability_of_1'] = proba[:,1]
        sessions_info.to_csv('predict_data/predictions.csv', index = False)

if __name__ == '__main__':
    main()
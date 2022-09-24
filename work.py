import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def get_season(month):
    month2season = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 4,
        12: 1
    }
    return month2season[month]

def make_dataset(group_df, cu_df):
    df_new = group_df.merge(
        cu_df[['Дата', 'Цена']], 
        how='left', 
        left_on='posting_date', 
        right_on='Дата')\
        .fillna(method='ffill')\
        .drop(columns=['Дата']).rename(columns={'Цена': 'cu_price'})

    df_new['month'] = df_new.posting_date.dt.month
    df_new['season'] = df_new['month'].apply(get_season)
    df_new['day'] = df_new.posting_date.dt.day
    df_new['day_of_week'] = df_new.posting_date.dt.day_of_week
    df_new['week'] = df_new.posting_date.dt.weekofyear
    df_new['year'] = df_new.posting_date.dt.year

    # df_new['april_or_march'] = ((df_new['month'] == 3) | (df_new['month'] == 4)).astype(int)

    df_new['month_before'] = df_new['posting_date'] - pd.DateOffset(days=30)
    df_new['2_month_before'] = df_new['posting_date'] - pd.DateOffset(days=60)

    df_new['out_bmu_sum_30d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(30).sum())
    
    df_new['out_bmu_mean_30d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(30).mean())
    
    df_new['out_bmu_max_30d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(30).max())

    df_new['cu_price_mean_30d'] = df_new['cu_price']\
        .transform(lambda x: x.rolling(30).mean())
    
    df_new['cu_price_max_30d'] = df_new['cu_price']\
        .transform(lambda x: x.rolling(30).max())
    
    df_new['cu_price_min_30d'] = df_new['cu_price']\
        .transform(lambda x: x.rolling(30).min())

    df_new['out_bmu_fact_count_sum_30d'] = df_new['out_bmu_fact_count']\
        .transform(lambda x: x.rolling(30).sum())
    
    df_new['out_bmu_fact_count_mean_30d'] = df_new['out_bmu_fact_count']\
        .transform(lambda x: x.rolling(30).mean())
    
    df_new['out_bmu_fact_count_max_30d'] = df_new['out_bmu_fact_count']\
        .transform(lambda x: x.rolling(30).max())

    df_new['out_bmu_sum_60d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(60).sum())
    
    df_new['out_bmu_mean_60d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(60).mean())
    
    df_new['out_bmu_max_60d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(60).max())

    df_new['out_bmu_sum_90d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(90).sum())
    
    df_new['out_bmu_mean_90d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(90).mean())

    df_new['out_bmu_max_90d'] = df_new['out_bmu_sum']\
        .transform(lambda x: x.rolling(90).max())

    copy_df_new = df_new[['posting_date', 'out_bmu_sum_30d', 'out_bmu_mean_30d', 'out_bmu_max_30d']].copy()
    copy_df_new.rename(columns={
        'out_bmu_sum_30d': 'out_bmu_sum_31_60d', 
        'out_bmu_mean_30d': 'out_bmu_mean_31_60d',
        'out_bmu_max_30d': 'out_bmu_max_31_60d',
        'posting_date': 'date'}, 
        inplace=True)
    df_new = pd.merge(df_new, copy_df_new, left_on='month_before', right_on='date').drop(columns=['date'])

    copy_df_new.rename(columns={
        'out_bmu_sum_31_60d': 'out_bmu_sum_61_90d',
        'out_bmu_mean_31_60d': 'out_bmu_mean_61_90d',
        'out_bmu_max_31_60d': 'out_bmu_max_61_90d', 
        'posting_date': 'date'},
        inplace=True)
    df_new = pd.merge(df_new, copy_df_new, left_on='month_before', right_on='date')\
        .drop(columns=['date'])

    # df_new = pd.merge(df_new, group_targets_df, left_on='posting_date', right_on='cut_date')

    df_new = df_new.drop(columns=['month_before', '2_month_before'])
    features = df_new.columns[8:]

    df_new = df_new.dropna()[features]

    return df_new

def predict(hackathon_test_df):
    # 
    cu_df = pd.read_csv('cu_norm.csv', sep=';')
    cu_df['Дата'] = pd.to_datetime(cu_df['Дата'])
    
    preds_data = {
        'sintez_group': [],
        **{f'predict_{i}': [] for i in range(12)}
    }
    
    for group in hackathon_test_df.sintez_group.unique():
        group_df = hackathon_test_df\
            .loc[hackathon_test_df.sintez_group == group, hackathon_test_df.columns[:7]]\
            .copy()
        group_df = make_dataset(group_df, cu_df)

        model = pickle.load(open(f'models/{group}_train_9.csv.pkl','rb'))
        model_preds = model.predict(group_df)

        preds_data['sintez_group'].append(group)
        for i in range(12):
            preds_data[f'predict_{i}'].append(model_preds[i, -1])
    
    preds = pd.DataFrame.from_dict(preds_data)
    return preds
import pandas as pd

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Nạp biến môi trường từ tệp ..env
load_dotenv()

# Truy cập các biến môi trường
db_uri = os.getenv('db_uri')
engine = create_engine(db_uri)

excel_file_knn = os.getenv('excel_file_knn')

def getData():
    df_feature = pd.read_sql_query('SELECT * FROM FEATURE', engine)
    df_product_feature = pd.read_sql_query('SELECT * FROM FEATURE_DETAIL', engine)
    df_product_feature = df_product_feature.join(df_feature.set_index('FEATURE_ID'), on='FEATURE_ID')
    df_feature_type = pd.read_sql_query('SELECT * FROM FEATURE_TYPE', engine)
    count_feature_type = []
    importance_max = []
    id = []

    for index, row in df_feature_type.iterrows():
        count = f"EXEC count_feature_type @featureTypeId = {row['FEARTURE_TYPE_ID']}"
        db_count = pd.read_sql_query(count, engine)
        for index1, row1 in db_count.iterrows():
            count_feature_type.append(row1['countFeatureType'])
            importance_max.append(row1['IMPORTANCE_MAX'])
        id.append(row['FEARTURE_TYPE_ID'])

    print("count_feature_type: ", count_feature_type)
    df_product_features = df_product_feature.pivot_table(
        index='UNIT_DETAIL_ID',
        columns='FEARTURE_TYPE_ID',
        values='IMPORTANCE'
    ).fillna(0)

    # print("111 df_product_features: ", df_product_feature.values[1], df_product_feature.values[1][4])
    # print(df_product_feature.columns)
    # print("df_product_features: ", df_product_features, df_product_features.columns)
    # print("i, j", len(df_product_features), len(count_feature_type))
    # print("id: ", id)
    for i in range(0, len(df_product_features)):
        for j in range(0, len(id)):
            # print("df_product_features.values[i][j] - importance_max[j]",
            #       df_product_features.values[i][j], importance_max[j], 1/importance_max[j],
            #       df_product_features.values[i][j] * 1/importance_max[j])
            df_product_features.values[i][j] = round(df_product_features.values[i][j] * 1/importance_max[j], 2)

    # df_product_features_has_label = df_product_features
    # categoryId = []
    # for unit_detail_id in df_product_features.index:
    #     newunit_detail = f"EXEC getCategoryIdId @unitDetailId = {unit_detail_id}"
    #     newunit_detail_db = pd.read_sql_query(newunit_detail, engine)
    #     for index, row in newunit_detail_db.iterrows():
    #         categoryId.append(row['CATEGORY_ID'])
    # df_product_features_has_label['label'] = categoryId
    #
    # print("4444444444", df_product_features_has_label)

    # Xuất DataFrame vào tệp Excel
    df_product_features.to_excel(excel_file_knn, index=True)

    print(f"DataFrame đã được xuất vào {excel_file_knn}")

    return df_product_features
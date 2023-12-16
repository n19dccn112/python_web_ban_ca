import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from data import getData
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

load_dotenv()
db_uri = os.getenv('db_uri')
engine = create_engine(db_uri)


def make_recommendation(unit_detail_id):
    # query = f"EXEC knn_data @productId = {unit_detail_id}"
    # db = pd.read_sql_query(query, engine)
    # list_specific = []
    # list_ft = []
    # for index, row in db.iterrows():
    #     list_specific.append(row['SPECIFIC'])
    #
    # df_unit_detail = pd.read_sql_query('SELECT * FROM UNIT_DETAIL', engine)
    # df_unit_detail = df_unit_detail.fillna(0)
    #
    # df_unit_detail_features = getData()
    # X = df_unit_detail_features
    #
    # # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    # X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    #
    # # print(df_unit_detail_features)
    # unit_details_to_idx = {
    #     unit_detail: i for i, unit_detail in
    #     enumerate(list(df_unit_detail_features.index))
    # }
    # # print("enumerate(list(df_unit_detail.set_index('id'))): ", enumerate(list(df_unit_detail.set_index('id'))))
    # print("unit_details: ", unit_details_to_idx)
    #
    # knn_model = NearestNeighbors(n_neighbors=20, metric='euclidean')
    # knn_model.fit(X)
    # Tìm kiếm các sản phẩm gần nhất
    # Tiếp theo là một đoạn mã tương tự khi bạn sử dụng mô hình KNN
    # distances, indices = knn_model.kneighbors(query_product)
    # print("distances, indices: ", distances, indices)
    #
    # # Hiển thị các sản phẩm gần nhất
    # nearest_products = merged_data.iloc[indices[0]]
    #
    # print("nearest_products: ", nearest_products)
    # print("nearest_products1: ", merged_data.iloc[indices[1]])
    # print("nearest_products2: ", merged_data.iloc[indices[2]])
    #
    # # Giả sử df là DataFrame bạn muốn chuyển thành tập hợp
    # df_list = list(nearest_products['PRODUCT_ID'].to_numpy())
    # print("df_set: ", df_list)
    #
    # unique_ordered_set = list(OrderedDict.fromkeys(df_list))
    # print("unique_ordered_set: ", unique_ordered_set)
    # list_product = [{'productId': int(x)} if isinstance(x, np.int64) else x for x in unique_ordered_set]
    # filtered_list = list(filter(lambda x: x != {'productId': int(product_id)}, list_product))
    # if len(filtered_list) > 6:
    #     filtered_list = filtered_list[:6]

    # print(filtered_list)
    # return filtered_list
    return []

# make_recommendation('1')




import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from typing_extensions import OrderedDict
from dotenv import load_dotenv

# Nạp biến môi trường từ tệp ..env
load_dotenv()
db_uri = os.getenv('db_uri')
engine = create_engine(db_uri)

def predict_model_knn2(product_id):
    query = f"EXEC knn_data @productId = {product_id}"
    db = pd.read_sql_query(query, engine)
    list_specific = []
    list_ft = []
    for index, row in db.iterrows():
        list_specific.append(row['SPECIFIC'])
        list_ft.append(row['FEARTURE_TYPE_ID'])

    # Kết hợp các bảng để tạo bảng chung
    merged_data = pd.read_sql_query('SET NOCOUNT ON; EXEC knn_data_X', engine)

    features = merged_data[['SPECIFIC', 'FEARTURE_TYPE_ID']]
    # print("merged_data: ", merged_data[['FEARTURE_TYPE_ID', 'SPECIFIC']])

    # Chuyển các giá trị chuỗi thành số
    label_encoder_spec = LabelEncoder()
    label_encoder_ft = LabelEncoder()
    features.loc[:, 'SPECIFIC'] = label_encoder_spec.fit_transform(features['SPECIFIC'])
    # print("features.loc[:, 'SPECIFIC']: ", merged_data[['SPECIFIC', 'FEARTURE_TYPE_ID']], features.loc[:, 'SPECIFIC'])

    features.loc[:, 'FEARTURE_TYPE_ID'] = label_encoder_ft.fit_transform(features['FEARTURE_TYPE_ID'])
    # print("features.loc[:, 'FEARTURE_TYPE_ID']: ", features.loc[:, 'FEARTURE_TYPE_ID'])

    # Tạo ma trận đặc trưng
    X = features[['SPECIFIC', 'FEARTURE_TYPE_ID']]
    print("X: ", X)

    # Huấn luyện mô hình KNN
    knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
    knn.fit(X)

    # list_specific = ['đuôi tam giác', 'sọc']
    # Chọn sản phẩm để tìm kiếm các sản phẩm gần nhất
    query_product = pd.DataFrame({
        'SPECIFIC': list_specific,
        'FEARTURE_TYPE_ID': list_ft
    })
    # Chuyển giá trị của 'SPECIFIC' cho sản phẩm cần tìm kiếm thành số
    query_product['SPECIFIC'] = label_encoder_spec.transform(query_product['SPECIFIC'])
    query_product['FEARTURE_TYPE_ID'] = label_encoder_ft.transform(query_product['FEARTURE_TYPE_ID'])
    # print("query_product['SPECIFIC']: ", query_product['SPECIFIC'])
    # print("query_product['FEARTURE_TYPE_ID']: ", query_product['FEARTURE_TYPE_ID'])

    print("query_product: ", query_product)
    # Tìm kiếm các sản phẩm gần nhất
    # Tiếp theo là một đoạn mã tương tự khi bạn sử dụng mô hình KNN
    distances, indices = knn.kneighbors(query_product)
    print("distances, indices: ", distances, indices)

    # Hiển thị các sản phẩm gần nhất
    nearest_products = merged_data.iloc[indices[0]]

    print("nearest_products: ", nearest_products)
    print("nearest_products1: ", merged_data.iloc[indices[1]])
    print("nearest_products2: ", merged_data.iloc[indices[2]])

    # Giả sử df là DataFrame bạn muốn chuyển thành tập hợp
    df_list = list(nearest_products['PRODUCT_ID'].to_numpy())
    print("df_set: ", df_list)

    unique_ordered_set = list(OrderedDict.fromkeys(df_list))
    print("unique_ordered_set: ", unique_ordered_set)
    list_product = [{'productId': int(x)} if isinstance(x, np.int64) else x for x in unique_ordered_set]
    filtered_list = list(filter(lambda x: x != {'productId': int(product_id)}, list_product))
    if len(filtered_list) > 6:
        filtered_list = filtered_list[:6]

    # print(filtered_list)
    return filtered_list


# predict_model_knn(['đuôi tam giác', 'sọc'])





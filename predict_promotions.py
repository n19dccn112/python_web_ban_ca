import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

# Nạp biến môi trường từ tệp ..env
load_dotenv()


# Truy cập các biến môi trường
db_uri = os.getenv('db_uri')
engine = create_engine(db_uri)
excel_file = os.getenv('excel_file')


def read_data():
    db = pd.read_sql_query('SET NOCOUNT ON; EXEC GetTable', engine)

    data = {}
    unit_detail_id = []
    product_price = []
    standard_price = []
    price_ship = []
    pond_amount = []
    amount_benh = []
    amount_chet = []
    amount_song = []
    amount_benh_1thang = []
    amount_chet_1thang = []
    amount_song_1thang = []
    daily_price = []
    stock_days = []

    for index, row in db.iterrows():
        unit_detail_id.append(row['UNIT_DETAIL_ID'])
        product_price.append(row['PRODUCT_PRICE'])
        standard_price.append(row['STANDARD_PRICE'])
        price_ship.append(row['PRICE_SHIP'])

        if pd.isnull(row['POND_AMOUNT']):
            pond_amount.append(0)
        else:
            pond_amount.append(row['POND_AMOUNT'])

        if pd.isnull(row['AMOUNT_BENH']):
            amount_benh.append(0)
        else:
            amount_benh.append(row['AMOUNT_BENH'])

        if pd.isnull(row['AMOUNT_CHET']):
            amount_chet.append(0)
        else:
            amount_chet.append(row['AMOUNT_CHET'])

        if pd.isnull(row['AMOUNT_SONG']):
            amount_song.append(0)
        else:
            amount_song.append(row['AMOUNT_SONG'])

        if pd.isnull(row['AMOUNT_BENH_1THANG']):
            amount_benh_1thang.append(0)
        else:
            amount_benh_1thang.append(row['AMOUNT_BENH_1THANG'])

        if pd.isnull(row['AMOUNT_CHET_1THANG']):
            amount_chet_1thang.append(0)
        else:
            amount_chet_1thang.append(row['AMOUNT_CHET_1THANG'])

        if pd.isnull(row['AMOUNT_SONG_1THANG']):
            amount_song_1thang.append(0)
        else:
            amount_song_1thang.append(row['AMOUNT_SONG_1THANG'])

        if pd.isnull(row['DAILY_PRICE']):
            daily_price.append(0)
        else:
            daily_price.append(row['DAILY_PRICE'])

        if pd.isnull(row['STOCK_DAYS']):
            stock_days.append(0)
        else:
            stock_days.append(row['STOCK_DAYS'])

    data['PRODUCT_PRICE'] = product_price
    data['STANDARD_PRICE'] = standard_price
    data['PRICE_SHIP'] = price_ship
    data['POND_AMOUNT'] = pond_amount
    data['AMOUNT_BENH'] = amount_benh
    data['AMOUNT_CHET'] = amount_chet
    data['AMOUNT_SONG'] = amount_song
    data['AMOUNT_BENH_1THANG'] = amount_benh_1thang
    data['AMOUNT_CHET_1THANG'] = amount_chet_1thang
    data['AMOUNT_SONG_1THANG'] = amount_song_1thang
    data['DAILY_PRICE'] = daily_price
    data['STOCK_DAYS'] = stock_days
    return data, unit_detail_id


def data_to_excel():
    data, unit_detail_id = read_data()
    df = pd.DataFrame(data).reset_index(drop=True)

    # Ghi DataFrame vào file Excel
    df.to_excel(excel_file, index=False)

    print(f'DataFrame đã được ghi vào {excel_file}')


def model_x():
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(excel_file)

    # Hiển thị một số dòng đầu tiên của dữ liệu
    print(df.head())

    # Xác định biến đầu ra (y)
    y_column_name = 'y'
    y = df[y_column_name]

    # Xác định biến đầu vào (X) bằng cách chọn một tập hợp các cột
    x_column_names = ['PRODUCT_PRICE', 'STANDARD_PRICE', 'PRICE_SHIP', 'POND_AMOUNT', 'AMOUNT_BENH', 'AMOUNT_CHET',
        'AMOUNT_SONG', 'AMOUNT_BENH_1THANG', 'AMOUNT_CHET_1THANG', 'AMOUNT_SONG_1THANG', 'DAILY_PRICE', 'STOCK_DAYS']
    x = df[x_column_names]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Xây dựng mô hình Linear Regression
    model = LinearRegression()

    # Huấn luyện mô hình trên tập huấn luyện
    model.fit(X_train, y_train)

    # Lưu mô hình vào file
    joblib.dump(model, 'model_linear_regression.joblib')


def predict_y(data, unit_detail_id):
    # Đọc mô hình từ file
    loaded_model = joblib.load('model_linear_regression.joblib')

    product_price = data['PRODUCT_PRICE']
    standard_price = data['STANDARD_PRICE']
    price_ship = data['PRICE_SHIP']
    pond_amount = data['POND_AMOUNT']
    amount_benh = data['AMOUNT_BENH']
    amount_chet = data['AMOUNT_CHET']
    amount_song = data['AMOUNT_SONG']
    amount_benh_1thang = data['AMOUNT_BENH_1THANG']
    amount_chet_1thang = data['AMOUNT_CHET_1THANG']
    amount_song_1thang = data['AMOUNT_SONG_1THANG']
    daily_price = data['DAILY_PRICE']
    stock_days = data['STOCK_DAYS']

    KQ = {}
    for i in range(len(unit_detail_id)):
        new_data = pd.DataFrame({
            'PRODUCT_PRICE': product_price[i],
            'STANDARD_PRICE': standard_price[i],
            'PRICE_SHIP': price_ship[i],
            'POND_AMOUNT': pond_amount[i],
            'AMOUNT_BENH': amount_benh[i],
            'AMOUNT_CHET': amount_chet[i],
            'AMOUNT_SONG': amount_song[i],
            'AMOUNT_BENH_1THANG': amount_benh_1thang[i],
            'AMOUNT_CHET_1THANG': amount_chet_1thang[i],
            'AMOUNT_SONG_1THANG': amount_song_1thang[i],
            'DAILY_PRICE': daily_price[i],
            'STOCK_DAYS': stock_days[i]
        }, index=[0])

        new_prediction = loaded_model.predict(new_data)
        if 0 < new_prediction <= 100:
            KQ[str(unit_detail_id[i])] = int(np.round(new_prediction))

    return KQ


def get_all_promotions():
    print("Running get_all_promotions at:", datetime.now())
    data, unit_detail_id = read_data()
    my_predict = predict_y(data, unit_detail_id)
    print("data")
    print("PRODUCT_PRICE", data["PRODUCT_PRICE"])
    print("STANDARD_PRICE", data["STANDARD_PRICE"])
    print("PRICE_SHIP", data["PRICE_SHIP"])
    print("POND_AMOUNT", data["POND_AMOUNT"])
    print("AMOUNT_BENH", data["AMOUNT_BENH"])
    print("AMOUNT_CHET", data["AMOUNT_CHET"])
    print("AMOUNT_SONG", data["AMOUNT_SONG"])
    print("AMOUNT_BENH_1THANG", data["AMOUNT_BENH_1THANG"])
    print("AMOUNT_CHET_1THANG", data["AMOUNT_CHET_1THANG"])
    print("AMOUNT_SONG_1THANG", data["AMOUNT_SONG_1THANG"])
    print("DAILY_PRICE", data["DAILY_PRICE"])
    print("STOCK_DAYS", data["STOCK_DAYS"])
    print("my_predict (id_unit_detail, phan_tram_khuyen_mai)", my_predict)
    np.savez('predict_promotions.npz', **my_predict)


def read_predict():
    loaded_data_my_predict = np.load('predict_promotions.npz')

    # Chuyển đổi dữ liệu từ định dạng NumPy array sang dict
    loaded_dict_my_predict = {int(key): int(value) for key, value in loaded_data_my_predict.items()}

    # In ra dữ liệu đã đọc được
    return loaded_dict_my_predict





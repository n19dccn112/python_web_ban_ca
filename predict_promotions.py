import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import pickle

# Nạp biến môi trường từ tệp ..env
load_dotenv()


# Truy cập các biến môi trường
db_uri = os.getenv('db_uri')
engine = create_engine(db_uri)
excel_file = os.getenv('excel_file')
excel_input_predict = os.getenv('excel_input_predict')


def tb_input(product_price, standard_price, price_ship, pond_amount, amount_benh,
             amount_chet, amount_song, sale_amount_1thang, amount_benh_1thang, amount_chet_1thang, amount_song_1thang,
             daily_price, stock_days):
    tb_product_price = int((sum(product_price)/len(product_price))/1000)*1000
    tb_standard_price = int((sum(standard_price)/len(standard_price))/1000)*1000
    tb_price_ship = int((sum(price_ship)/len(price_ship))/1000)*1000
    tb_pond_amount = round(sum(pond_amount)/len(pond_amount))
    tb_amount_benh = round(sum(amount_benh)/len(amount_benh))
    tb_amount_chet = round(sum(amount_chet)/len(amount_chet))
    tb_amount_song = round(sum(amount_song)/len(amount_song))
    tb_sale_amount_1thang = round(sum(sale_amount_1thang)/len(sale_amount_1thang))
    tb_amount_benh_1thang = round(sum(amount_benh_1thang)/len(amount_benh_1thang))
    tb_amount_chet_1thang = round(sum(amount_chet_1thang)/len(amount_chet_1thang))
    tb_amount_song_1thang = round(sum(amount_song_1thang)/len(amount_song_1thang))
    tb_daily_price = int((sum(daily_price)/len(daily_price))/100)*100
    tb_stock_days = round(sum(stock_days)/len(stock_days))

    min_product_price = min(product_price)
    min_standard_price = min(standard_price)
    min_price_ship = min(price_ship)
    min_pond_amount = min(pond_amount)
    min_amount_benh = min(amount_benh)
    min_amount_chet = min(amount_chet)
    min_amount_song = min(amount_song)
    min_sale_amount_1thang = min(sale_amount_1thang)
    min_amount_benh_1thang = min(amount_benh_1thang)
    min_amount_chet_1thang = min(amount_chet_1thang)
    min_amount_song_1thang = min(amount_song_1thang)
    min_daily_price = min(daily_price)
    min_stock_days = min(stock_days)

    max_product_price = max(product_price)
    max_standard_price = max(standard_price)
    max_price_ship = max(price_ship)
    max_pond_amount = max(pond_amount)
    max_amount_benh = max(amount_benh)
    max_amount_chet = max(amount_chet)
    max_amount_song = max(amount_song)
    max_sale_amount_1thang = max(sale_amount_1thang)
    max_amount_benh_1thang = max(amount_benh_1thang)
    max_amount_chet_1thang = max(amount_chet_1thang)
    max_amount_song_1thang = max(amount_song_1thang)
    max_daily_price = max(daily_price)
    max_stock_days = max(stock_days)

    sd_product_price = max(max_product_price-tb_product_price, tb_product_price-min_product_price)
    sd_standard_price = max(max_standard_price-tb_standard_price, tb_standard_price-min_standard_price)
    sd_price_ship = max(max_price_ship-tb_price_ship, tb_price_ship-min_price_ship)
    sd_pond_amount = max(max_pond_amount-tb_pond_amount, tb_pond_amount-min_pond_amount)
    sd_amount_benh = max(max_amount_benh-tb_amount_benh, tb_amount_benh-min_amount_benh)
    sd_amount_chet = max(max_amount_chet-tb_amount_chet, tb_amount_chet-min_amount_chet)
    sd_amount_song = max(max_amount_song-tb_amount_song, tb_amount_song-min_amount_song)
    sd_sale_amount_1thang = max(max_sale_amount_1thang-tb_sale_amount_1thang, tb_sale_amount_1thang-min_sale_amount_1thang)
    sd_amount_benh_1thang = max(max_amount_benh_1thang-tb_amount_benh_1thang, tb_amount_benh_1thang-min_amount_benh_1thang)
    sd_amount_chet_1thang = max(max_amount_chet_1thang-tb_amount_chet_1thang, tb_amount_chet_1thang-min_amount_chet_1thang)
    sd_amount_song_1thang = max(max_amount_song_1thang-tb_amount_song_1thang, tb_amount_song_1thang-min_amount_song_1thang)
    sd_daily_price = max(max_daily_price-tb_daily_price, tb_daily_price-min_daily_price)
    sd_stock_days = max(max_stock_days-tb_stock_days, tb_stock_days-min_stock_days)

    with open(excel_input_predict, 'wb') as file:
        pickle.dump((tb_product_price, tb_standard_price, tb_price_ship, tb_pond_amount, tb_amount_benh,
                     tb_amount_chet, tb_amount_song, tb_sale_amount_1thang, tb_amount_benh_1thang, tb_amount_chet_1thang,
                     tb_amount_song_1thang, tb_daily_price, tb_stock_days,
                     sd_product_price, sd_standard_price, sd_price_ship, sd_pond_amount, sd_amount_benh,
                     sd_amount_chet, sd_amount_song, sd_sale_amount_1thang, sd_amount_benh_1thang, sd_amount_chet_1thang,
                     sd_amount_song_1thang, sd_daily_price, sd_stock_days), file)

    print("Giá bán trung bình:\t\t\t", tb_product_price, "\t\tmin - max: ", min_product_price, "-", max_product_price, "\t\t\tĐộ lệch chuẩn cho phép: ", sd_product_price)
    print("Giá mua trung bình:\t\t\t", tb_standard_price, "\t\tmin - max: ", round(min_standard_price, 2), "-", max_standard_price, "\t\t\tĐộ lệch chuẩn cho phép: ", sd_standard_price)
    print("Phí giao hàng:\t\t\t\t", tb_price_ship, "\t\tmin - max: ", min_price_ship, "\t-", max_price_ship, "\t\t\tĐộ lệch chuẩn cho phép: ", sd_price_ship)
    print("Số lượng tồn kho:\t\t\t", tb_pond_amount, "\t\tmin - max: ", min_pond_amount, "\t\t-", max_pond_amount, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_pond_amount)
    print("Số lượng bệnh:\t\t\t\t", tb_amount_benh, "\t\t\tmin - max: ", min_amount_benh, "\t\t-", max_amount_benh, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_amount_benh)
    print("Số lượng chết:\t\t\t\t", tb_amount_chet, "\t\t\tmin - max: ", min_amount_chet, "\t\t-", max_amount_chet, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_amount_chet)
    print("Số lượng sống:\t\t\t\t", tb_amount_song, "\t\tmin - max: ", min_amount_song, "\t\t-", max_amount_song, "\t\t\tĐộ lệch chuẩn cho phép: ", sd_amount_song)
    print("Số lượng bán trong 1 tháng:\t", tb_sale_amount_1thang, "\t\t\tmin - max: ", min_sale_amount_1thang, "\t\t-", max_sale_amount_1thang, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_sale_amount_1thang)
    print("Số lượng bệnh trong 1 tháng:", tb_amount_benh_1thang, "\t\t\tmin - max: ", min_amount_benh_1thang, "\t\t-", max_amount_benh_1thang, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_amount_benh_1thang)
    print("Số lượng chết trong 1 tháng:", tb_amount_chet_1thang, "\t\t\tmin - max: ", min_amount_chet_1thang, "\t\t-", max_amount_chet_1thang, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_amount_chet_1thang)
    print("Số lượng sống trong 1 tháng:", tb_amount_song_1thang, "\t\tmin - max: ", min_amount_song_1thang, "\t\t-", max_amount_song_1thang, "\t\t\tĐộ lệch chuẩn cho phép: ", sd_amount_song_1thang)
    print("Chi phí hằng ngày:\t\t\t", tb_daily_price, "\t\tmin - max: ", min_daily_price, "\t-", max_daily_price, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_daily_price)
    print("Ngày tồn kho:\t\t\t\t", tb_stock_days, "\t\tmin - max: ", min_stock_days, "\t\t-", max_stock_days, "\t\t\t\tĐộ lệch chuẩn cho phép: ", sd_stock_days)


def filter_data(row, tb, sd):
    if pd.isnull(row):
        row = 0
    # print(tb-sd, "<=", row, "<=", tb+sd, tb-sd <= row <= tb+sd)
    if tb-sd <= row <= tb+sd:
        return 0
    return 1


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
    sale_amount_1thang = []
    amount_benh_1thang = []
    amount_chet_1thang = []
    amount_song_1thang = []
    daily_price = []
    stock_days = []

    with open(excel_input_predict, 'rb') as file:
        (tb_product_price, tb_standard_price, tb_price_ship, tb_pond_amount, tb_amount_benh,
         tb_amount_chet, tb_amount_song, tb_sale_amount_1thang, tb_amount_benh_1thang, tb_amount_chet_1thang,
         tb_amount_song_1thang, tb_daily_price, tb_stock_days,
         sd_product_price, sd_standard_price, sd_price_ship, sd_pond_amount, sd_amount_benh,
         sd_amount_chet, sd_amount_song, sd_sale_amount_1thang, sd_amount_benh_1thang, sd_amount_chet_1thang,
         sd_amount_song_1thang, sd_daily_price, sd_stock_days) = pickle.load(file)

    for index, row in db.iterrows():
        if (filter_data(row['PRODUCT_PRICE'], tb_product_price, sd_product_price) == 1 or
                filter_data(row['STANDARD_PRICE'], tb_standard_price, sd_standard_price) == 1 or
                filter_data(row['PRICE_SHIP'], tb_price_ship, sd_price_ship) == 1 or
                filter_data(row['POND_AMOUNT'], tb_pond_amount, sd_pond_amount) == 1 or
                filter_data(row['AMOUNT_BENH'], tb_amount_benh, sd_amount_benh) == 1 or
                filter_data(row['AMOUNT_CHET'], tb_amount_chet, sd_amount_chet) == 1 or
                filter_data(row['AMOUNT_SONG'], tb_amount_song, sd_amount_song) == 1 or
                filter_data(row['SALE_AMOUNT_1THANG'], tb_sale_amount_1thang, sd_sale_amount_1thang) == 1 or
                filter_data(row['AMOUNT_BENH_1THANG'], tb_amount_benh_1thang, sd_amount_benh_1thang) == 1 or
                filter_data(row['AMOUNT_CHET_1THANG'], tb_amount_chet_1thang, sd_amount_chet_1thang) == 1 or
                filter_data(row['AMOUNT_SONG_1THANG'], tb_amount_song_1thang, sd_amount_song_1thang) == 1 or
                filter_data(row['DAILY_PRICE'], tb_daily_price, sd_daily_price) == 1 or
                filter_data(row['STOCK_DAYS'], tb_stock_days, sd_stock_days) == 1):
            print("bỏ qua")
        else:
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

            if pd.isnull(row['SALE_AMOUNT_1THANG']):
                sale_amount_1thang.append(0)
            else:
                sale_amount_1thang.append(row['SALE_AMOUNT_1THANG'])

            if pd.isnull(row['AMOUNT_BENH_1THANG']) or row['AMOUNT_BENH_1THANG'] <= 0:
                amount_benh_1thang.append(0)
            else:
                amount_benh_1thang.append(row['AMOUNT_BENH_1THANG'])

            if pd.isnull(row['AMOUNT_CHET_1THANG']) or row['AMOUNT_CHET_1THANG'] <= 0:
                amount_chet_1thang.append(0)
            else:
                amount_chet_1thang.append(row['AMOUNT_CHET_1THANG'])

            if pd.isnull(row['AMOUNT_SONG_1THANG']) or row['AMOUNT_SONG_1THANG'] <= 0:
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

    # tb_input(product_price, standard_price, price_ship, pond_amount, amount_benh,
    #          amount_chet, amount_song, sale_amount_1thang, amount_benh_1thang, amount_chet_1thang, amount_song_1thang,
    #          daily_price, stock_days)

    data['PRODUCT_PRICE'] = product_price
    data['STANDARD_PRICE'] = standard_price
    data['PRICE_SHIP'] = price_ship
    data['POND_AMOUNT'] = pond_amount
    data['AMOUNT_BENH'] = amount_benh
    data['AMOUNT_CHET'] = amount_chet
    data['AMOUNT_SONG'] = amount_song
    data['SALE_AMOUNT_1THANG'] = sale_amount_1thang
    data['AMOUNT_BENH_1THANG'] = amount_benh_1thang
    data['AMOUNT_CHET_1THANG'] = amount_chet_1thang
    data['AMOUNT_SONG_1THANG'] = amount_song_1thang
    data['DAILY_PRICE'] = daily_price
    data['STOCK_DAYS'] = stock_days
    print("data:", data['SALE_AMOUNT_1THANG'])
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
        'AMOUNT_SONG', 'SALE_AMOUNT_1THANG', 'AMOUNT_BENH_1THANG', 'AMOUNT_CHET_1THANG', 'AMOUNT_SONG_1THANG', 'DAILY_PRICE', 'STOCK_DAYS']
    x = df[x_column_names]

    # Xây dựng mô hình Linear Regression
    model = LinearRegression()

    # Huấn luyện mô hình trên tập huấn luyện
    model.fit(x, y)

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
    amount_sale_1thang = data['SALE_AMOUNT_1THANG']
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
            'SALE_AMOUNT_1THANG': amount_sale_1thang[i],
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
    # print("PRODUCT_PRICE", data["PRODUCT_PRICE"])
    # print("STANDARD_PRICE", data["STANDARD_PRICE"])
    # print("PRICE_SHIP", data["PRICE_SHIP"])
    # print("POND_AMOUNT", data["POND_AMOUNT"])
    # print("AMOUNT_BENH", data["AMOUNT_BENH"])
    # print("AMOUNT_CHET", data["AMOUNT_CHET"])
    # print("AMOUNT_SONG", data["AMOUNT_SONG"])
    # print("SALE_AMOUNT_1THANG", data["SALE_AMOUNT_1THANG"])
    # print("AMOUNT_BENH_1THANG", data["AMOUNT_BENH_1THANG"])
    # print("AMOUNT_CHET_1THANG", data["AMOUNT_CHET_1THANG"])
    # print("AMOUNT_SONG_1THANG", data["AMOUNT_SONG_1THANG"])
    # print("DAILY_PRICE", data["DAILY_PRICE"])
    # print("STOCK_DAYS", data["STOCK_DAYS"])
    # print("my_predict (id_unit_detail, phan_tram_khuyen_mai)", my_predict)
    for i in range(len(unit_detail_id)):
        if i > 3:
            break
        try:
            print("Giá bán:\t\t\t", data["PRODUCT_PRICE"][i], ";\t\tGiá mua vào:\t\t\t\t", data["STANDARD_PRICE"][i],
            ";\nSố ngày tồn kho:\t", data["STOCK_DAYS"][i], ";\t\t\tPhí giao hàng:\t\t\t\t", data["PRICE_SHIP"][i],
            ";\nChi phí hằng ngày:\t", data["DAILY_PRICE"][i], ";\t\t\tSố lượng tồn kho:\t\t\t", data["POND_AMOUNT"][i],
            ";\nSố lượng bệnh:\t\t", data["AMOUNT_BENH"][i], ";\t\t\tSố lượng bệnh trong 1 tháng:", data["AMOUNT_BENH_1THANG"][i],
            ";\nSố lượng sống:\t\t", data["AMOUNT_SONG"][i], ";\t\t\tSố lượng sống trong 1 tháng:", data["AMOUNT_SONG_1THANG"][i],
            ";\nSố lượng chết:\t\t", data["AMOUNT_CHET"][i], ";\t\t\tSố lượng chết trong 1 tháng:", data["AMOUNT_CHET_1THANG"][i],
            ";\n\t\t\t\t\t\t\t\t\tSố lượng bán trong 1 tháng: ", data["SALE_AMOUNT_1THANG"][i],


            )
            print("Sản phẩm unitDetailid: ", unit_detail_id[i], ";\t\t\tphan_tram_khuyen_mai: ", my_predict[str(unit_detail_id[i])], "%")
            print("============================================================================================\n")
        except Exception as e:
            print("lỗi")


    np.savez('predict_promotions.npz', **my_predict)


def read_predict():
    loaded_data_my_predict = np.load('predict_promotions.npz')

    # Chuyển đổi dữ liệu từ định dạng NumPy array sang dict
    loaded_dict_my_predict = {int(key): int(value) for key, value in loaded_data_my_predict.items()}

    # In ra dữ liệu đã đọc được
    return loaded_dict_my_predict





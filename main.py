import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from flask import Flask, request, Response, jsonify

from knn import make_recommendation
# from knn2 import predict_model_knn2
from predict_promotions import read_predict, get_all_promotions, data_to_excel, model_x

from flask_restful import Api
from flask_cors import CORS
import threading
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

from search import search, data_knn

app = Flask(__name__)
api = Api(app)
cors = CORS(app)

# Nạp biến môi trường từ tệp ..env
load_dotenv()

# Truy cập các biến môi trường
db_uri = os.getenv('db_uri')
excel_file = os.getenv('excel_file')
print("db_uri: ", db_uri)
engine = create_engine(db_uri)


@app.route('/', methods=['GET'])
def home():
    return '<a href="http://localhost:3000/admin">Đăng nhập Admin</a>'


@app.route('/predictPromotion', methods=['GET'])
def get_predict_promotion():
    result_list = [{'unitDetailId': key, 'promotion': value} for key, value in read_predict().items()]

    # Trả về danh sách JSON
    return jsonify(result_list)


@app.route('/predictKnn', methods=['GET'])
def get_predict_knn():
    product_id = request.args.get('productId')
    print("==============9]]]]]]================")
    print(product_id)
    if product_id is None:
        return jsonify([])
    return jsonify(make_recommendation(product_id))


@app.route('/predictSearch', methods=['GET'])
def get_predict_search():
    query = request.args.get('query')
    print("=============000000000000000000000====")
    print(query)
    if query is None:
        return jsonify([])
    results = search(query)
    print("results: ", results)
    return jsonify(results)


# Tạo một scheduler
# Tạo một scheduler
scheduler = BlockingScheduler()

# Hẹn giờ chạy hàm vào 0 giờ tối mỗi ngày
scheduler.add_job(get_all_promotions, 'cron', hour=0)


def run_flask():
    print("chạy run_flask")
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    make_recommendation(1)
    # data_knn()
    # all_points = np.array([[42, 0],
    #                            [29, 0],
    #                            [45, 0],
    #                            [22, 0]])
    #
    # # Điểm query
    # query_point = np.array([17, 1])
    #
    # # Tính khoảng cách giữa query_point và tất cả các điểm trong all_points
    # distances = np.linalg.norm(all_points - query_point, axis=1)
    #
    # print("Khoảng cách đến mỗi điểm:")
    # print(distances)
    # predict_model_knn2(1)
    # data_to_excel()
    # model_x()
    # get_all_promotions()
    # Bắt đầu scheduler trong một luồng khác nhau
    scheduler_thread = threading.Thread(target=scheduler.start)
    scheduler_thread.start()
    print("start scheduler")

    # Chạy ứng dụng Flask trong một luồng khác nhau
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Đợi cả hai luồng hoàn thành
    scheduler_thread.join()
    flask_thread.join()




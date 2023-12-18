import math
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Nạp biến môi trường từ tệp ..env
load_dotenv()

# Truy cập các biến môi trường
db_uri = os.getenv('db_uri')
engine = create_engine(db_uri)

connection = engine.connect()


def dicttumadoc_dictmadocdodaidoc_dodaitb(doc_vitri_cau_dict):
    doc_word_list_vitri_dict = {}
    doc_vitridoc_lendoc_dict = {}
    for i, sentence in doc_vitri_cau_dict.items():
        sentence = sentence.lower()
        words = sentence.split()
        doc_vitridoc_lendoc_dict[i] = len(words)
        for word in words:
            if word not in doc_word_list_vitri_dict:
                doc_word_list_vitri_dict[word] = [i]
            else:
                doc_word_list_vitri_dict[word].append(i)
    sort_doc_word_list_vitri_dict = dict(sorted(doc_word_list_vitri_dict.items(), key=lambda x: x[0].lower()))
    tb_lendoc = sum(doc_vitridoc_lendoc_dict.values()) / len(doc_vitridoc_lendoc_dict)
    return sort_doc_word_list_vitri_dict, doc_vitridoc_lendoc_dict, tb_lendoc


def RSV_word(n, df, tf, dl, avdt, k=2, b=0.75):
    # n là số văn bản
    # df là số văn bản có từ qi
    # tf là số lần xuất hiện của từ trong tập đang xét
    # dl là độ dài vb hiện tại
    # avdt là độ dài tb tất cả vb
    # k và b là các tham số của BM25;
    # k điều khiển tỷ lệ tần số; b chuẩn hóa độ dài tài liệu
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    numerator = tf * (k + 1)
    denominator = tf + k * (1 - b + b * (dl / avdt))
    result = idf * (numerator / denominator)
    return result


def dicttulistvitri(query, doc_word_list_vitri_dict):
    query_words_list = [word.lower() for word in query.split()]
    query_tu_list_vitri_dict = {}
    for key, value in doc_word_list_vitri_dict.items():
        if key in query_words_list:
            query_tu_list_vitri_dict[key] = value
    return query_tu_list_vitri_dict


def rsv_bm25(doc_vitri_cau_dict, doc_vitridoc_lendoc_dict, tb_lendoc, query_tu_list_vitri_dict):
    vitridoc_rsv_dict = {}
    vitridoc_tu_dict = {}
    for i, sentence in doc_vitri_cau_dict.items():  # từng 1: D1, 2: D2, 3: D3,...
        c1 = 0
        dict_tu_rsv = []
        for tu, listvitri in query_tu_list_vitri_dict.items():  # Các từ của câu query q1, q2, q3,..
            tf = 0
            sentence = sentence.lower()
            words = sentence.split()
            for word in words:  # Duyệt qua các từ của D1: w1, w2, w3,...
                if tu.lower() == word.lower():
                    tf += 1
            if tf != 0:
                df = 0
                for i1, sentence1 in doc_vitri_cau_dict.items():
                    sentence1 = sentence1.lower()
                    words1 = sentence1.split()
                    break_for_1 = 0
                    for word1 in words1:
                        if tu == word1:
                            df += 1
                            break_for_1 = 1
                            break
                    if break_for_1 == 1:
                        break
                a = RSV_word(len(doc_vitri_cau_dict), df, tf, doc_vitridoc_lendoc_dict[i], tb_lendoc)
                c1 += a
                dict_tu_rsv.append(tu)
                dict_tu_rsv.append(a)
        if c1 != 0:
            vitridoc_rsv_dict[i] = c1
        vitridoc_tu_dict[i] = dict_tu_rsv
    return vitridoc_rsv_dict, vitridoc_tu_dict


def run_BM25(query, doc_vitri_cau_dict, list_id):
    # doc1: con co
    # doc2: co be
    # doc_word_list_vitri_dict: {con: [1], co: [1, 2], be: [2]}
    # doc_vitridoc_lendoc_dict: {1: 2, 2: 2}
    # tb_lendoc: 2
    doc_word_list_vitri_dict, doc_vitridoc_lendoc_dict, tb_lendoc = dicttumadoc_dictmadocdodaidoc_dodaitb(doc_vitri_cau_dict)
    # query: con be
    # query_tu_list_vitri_dict: {con: [1], be: [2]}
    query_tu_list_vitri_dict = dicttulistvitri(query, doc_word_list_vitri_dict)
    vitridoc_rsv_dict, vitridoc_tu_dict = rsv_bm25(doc_vitri_cau_dict, doc_vitridoc_lendoc_dict, tb_lendoc, query_tu_list_vitri_dict)
    vitridoc_rsv_dict = dict(sorted(vitridoc_rsv_dict.items(), key=lambda x: x[1], reverse=True))
    for id, rsv in vitridoc_rsv_dict.items():
        list_id.append(id)
        print("unitDetailId: ", id, "Độ tương đồng: ", rsv, vitridoc_tu_dict[id])

    return list_id, vitridoc_rsv_dict


def search_BM25(query):
    df = pd.read_sql_query('SELECT * FROM PRODUCT', engine)
    doc_vitri_cau_dict = {}
    list_id = []

    for index, row in df.iterrows():
        doc_vitri_cau_dict[row['PRODUCT_ID']] = row['PRODUCT_DESCRIPTION'] + row['PRODUCT_NAME']

    list_id, vitridoc_rsv_dict = run_BM25(query, doc_vitri_cau_dict, list_id)


    return list_id


def data_knn():
    doc_vitri_cau_dict = {}
    list_id = []
    df = pd.read_sql_query('SET NOCOUNT ON; EXEC data_X_knn', engine)
    for index, row in df.iterrows():
        doc_vitri_cau_dict[row['UNIT_DETAIL_ID']] = row['SPECIFIC']

    df = pd.read_sql_query('SELECT * FROM FEATURE', engine)
    featureId_specific_dict = {}

    for index, row in df.iterrows():
        featureId_specific_dict[row['FEATURE_ID']] = row['SPECIFIC']
        query = row['SPECIFIC']
        list_id, vitridoc_rsv_dict = run_BM25(query, doc_vitri_cau_dict, list_id)
        for vitri, rsv in vitridoc_rsv_dict.items():
            sql_command = text("EXEC update_feature @unitDetailId = :unitDetailId, @featureId = :featureId, @point = :point").bindparams(
                unitDetailId=vitri, featureId=row['FEATURE_ID'], point=round(rsv)
            )
            result = connection.execute(sql_command)
        connection.commit()
    connection.close()
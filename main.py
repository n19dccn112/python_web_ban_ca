from flask import Flask, request, Response, jsonify
from flask_restful import Api, reqparse
from flask_cors import CORS, cross_origin
from knn import make_recommendation
from search import search
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
api = Api(app)
cors = CORS(app)

@app.route('/', methods=['GET'])
def home():
    return \
        'recommender for webapp of D19CQQCNPM01'

@app.route('/recommender', methods=['GET'])
def getProducts():
    productId = request.args.get('productId')
    print("==============9]]]]]]================")
    print(productId)
    if productId is None:
        return Response([], mimetype='application/json')
    return Response(make_recommendation(productId), mimetype='application/json')

@app.route('/search', methods=['GET'])
def getSearch():
    query = request.args.get('query')
    print("=============000000000000000000000====")
    print(query)
    if query is None:
        return jsonify([])
    results = search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
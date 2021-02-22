import json
import pickle
import numpy as np
from flask import Flask, jsonify
from flask_restful import Api, Resource

class DecisionTree(Resource):
    def __init__(self):
        self.model = 'model.pkl'
        
    def get(self):
        with open(self.model, 'rb') as file:
            _model = pickle.load(file)
            
            data = np.random.rand(5, 13)
            return jsonify(np.array2string(_model.predict(data)))
            
app = Flask(__name__)
api = Api(app)

api.add_resource(DecisionTree, '/api')

if __name__ == '__main__':
    app.run(debug = True)
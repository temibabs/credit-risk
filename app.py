import pickle
import pandas as pd
import sklearn

from flask import Flask, request

app = Flask(__name__)

# Model loading
model = pickle.load(open("/Users/temi/PycharmProjects/credit-risk-api/best-model", 'rb'))


@app.route('/v1/score', methods=['POST'])
def scoring():  # put application's code here
    loan = request.get_json(force=True)

    try:
        df = pd.DataFrame.from_dict(loan, orient='index')
        print(df.T)
        prediction = model.predict_proba(df.T)
        print(prediction)
    except:
        return {'status': 'ERROR'}, 400

    return {'probability': prediction[0, 1]}, 200


if __name__ == '__main__':
    app.run()

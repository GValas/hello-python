import base64
import io

import keras
import numpy as np
import tensorflow as tf
from flask import Flask, current_app, request, jsonify
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image

model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None,
                                                    input_shape=None)
graph = tf.get_default_graph()
app = Flask(__name__)


def predict_me(image_file):
    img = image.load_img(image_file, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    global graph
    with graph.as_default():
        preds = model.predict(x)

    top3 = decode_predictions(preds, top=10)[0]

    predictions = [{'label': label, 'description': description, 'probability': probability * 100.0}
                   for label, description, probability in top3]
    return predictions





@app.route('/', methods=['POST'])
def predict():
    data = {}
    try:
        data = request.get_json()['data']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400

    data = base64.b64decode(data)

    image = io.BytesIO(data)
    #
    # from PIL import Image
    # Image.open(image).show()

    predictions = predict_me(image)
    current_app.logger.info('Predictions: %s', predictions)
    return jsonify(predictions=predictions)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080, debug=True)

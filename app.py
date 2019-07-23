import os

import tensorflow as tf
from tensorflow_hub import Module
from flask import Flask, jsonify, request
# import tf_sentencepiece

app = Flask(__name__)
print("Current position: ", os.path.dirname(os.path.abspath(__file__)))
print("Subdirectories")

d = '..'
print(
    [os.path.join(d, o) for o in os.listdir(d)
     if os.path.isdir(os.path.join(d, o))])


def load_model():
    """Load and return the model"""
    # module_url = os.path.join('..', 'input', 'model')  # 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/1'
    module_url = os.path.join('model_big')  # 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/1'

    # Set up graph.
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        multiling_embed = Module(module_url)
        embedded_text = multiling_embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Initialize session.
    session = tf.Session(graph=g)
    session.run(init_op)
    return session, embedded_text, text_input


# you can then reference this model object in evaluate function/handler
session, embedded_text, text_input = load_model()


# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body,
# including images, JSON, encoded-data, etc.)
@app.route('/', methods=["POST"])
def evaluate():
    """"Preprocessing the data and evaluate the model"""
    # TODO: data/input preprocessing
    req_data = request.get_json()
    text = req_data['text']
    # text = request.args.get('text')
    # eg: request.files.get('file')
    # eg: request.args.get('style')
    # eg: request.form.get('model_name')
    if not isinstance(text, list):
        text = [text]

    print(text)
    embedded_result = session.run(embedded_text, feed_dict={text_input: text})

    # TODO: return prediction
    return jsonify({'embedded_text': embedded_result.tolist()})


# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False, port=5001)

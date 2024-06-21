from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import pickle
import requests
from io import BytesIO

app = Flask(__name__)

# Load PCA Components and KNN Model
with open('final_pca_components2.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)

with open('final_knn_model2.pkl', 'rb') as knn_file:
    knn_model = pickle.load(knn_file)

# Load the pre-trained model
base_model = VGG16(include_top=False, input_shape=(256, 256, 3))
model = Sequential()
for layer in base_model.layers:
    model.add(layer)
model.add(GlobalAveragePooling2D())
model.summary()

# Function to Read and Process Image
def read_img_from_url(image_url):
    response = requests.get(image_url)
    image = load_img(BytesIO(response.content), target_size=(256, 256))
    image = img_to_array(image)
    image = image / 255.
    return image

@app.route('/similar_images', methods=['POST'])
def get_similar_images():
    if not request.is_json:
        return jsonify({'error': 'Invalid input'}), 400

    data = request.get_json()
    if 'img' not in data:
        return jsonify({'error': 'No image URL provided'}), 400

    image_url = data['img']
    try:
        image = read_img_from_url(image_url)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

    # Extract features from the image
    features = model.predict(np.array([image]))
    pca_features = pca.transform(features)

    # Find similar images using KNN
    dist, indices = knn_model.kneighbors(pca_features.reshape(1, -1))
    similar_ids = indices[0].tolist()

    return jsonify({'similar_ids': similar_ids})

if __name__ == '__main__':
    app.run(debug=True)

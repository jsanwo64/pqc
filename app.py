from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage import exposure

app = Flask(__name__)

def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

@app.route('/check_quality', methods=['POST'])
def check_quality():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    blur = calculate_blur(img)
    brightness = calculate_brightness(img)

    quality = "Good"
    if blur < 100:
        quality = "Blurry"
    elif brightness < 50:
        quality = "Too Dark"
    elif brightness > 200:
        quality = "Too Bright"

    return jsonify({
        'blur_score': round(blur, 2),
        'brightness_score': round(brightness, 2),
        'quality': quality
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

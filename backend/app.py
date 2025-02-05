from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_objects(image_path):
    img = cv2.imread(image_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    result_path = 'uploads/result.jpg'
    cv2.imwrite(result_path, img)
    
    return result_path

@app.route('/detect', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
        file.save(filepath)
        
        result_image_path = detect_objects(filepath)
        
        return jsonify({"message": "Object detection successful", "result_image_path": result_image_path}), 200

if __name__ == '__main__':
    app.run(debug=True)
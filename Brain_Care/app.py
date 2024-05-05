import io
import shutil

from PIL import Image
from flask import Flask, render_template, make_response, render_template_string, request, jsonify
import pdfkit
import os
import numpy as np
import tensorflow as tf
import cv2
from keras.src.applications.efficientnet import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
import numpy as np
from flask import Flask, jsonify, request
# from keras.models import load_model
# from keras.preprocessing import image
# from keras.applications.inception_v3 import preprocess_input

from PIL import Image
import io

app = Flask(__name__)

class_labels = {
    0: 'non-hemorrhage',
    1: 'hemorrhage'
}

# Load the Keras model
# model = load_model('model.h5')



from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class_labels = {
    0: 'non-hemorrhage',
    1: 'hemorrhage'
}



# Load the Keras model
# model = load_model('model.h5')

# HTML template for the PDF report
html_template = """
<html>
<head>
    <title>Patient Report</title>
</head>
<body>
    <h1>Patient Report</h1>
    <p><strong>Name:</strong> {{ name }}</p>
    <p><strong>Age:</strong> {{ age }}</p>
    <p><strong>Symptoms:</strong> {{ symptoms }}</p>
    <p><strong>Gender:</strong>{{ gender }}</p>
    <p><strong>Diagnosis:</strong> {{ diagnosis }}</p>
    <h2>Prediction:</h2>
    <p>{{ disease }}</p>
    <img src="{{ detection }}" alt="image not found" width="400" height="200">
    <img src="{{ mask }}" alt="image not found" width="400" height="200">
</body>
</html>
"""

# @app.route('/predict', methods=['POST'])
# def predict():

@app.route('/generate_report', methods=['POST'])
def generate_report():
    current_directory = os.path.dirname(__file__)

    # Get the image file from the request
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    symptoms = request.form['symptoms']
    percentage = request.form['percentage']
    disease = request.form['disease']
    print("Name:", name)
    print("Age:", age)
    print("Gender:", gender)
    print("Symptoms:", symptoms)
    print("percentag",percentage)
    print("disease",disease)
    ct_image = request.files['file'];


    # Print form data


    # Save the image file to the directory of app.py
    file_path = os.path.join(current_directory, ct_image.filename)
    ct_image.save(file_path)
    print(file_path)
    image_path1 = file_path

    # Run hemorrhage detection on both images
    urls = detect_hemorrhage(image_path1)
    print(urls)

    patient_data = {
        "name": name,
        "age": age,
        "diagnosis": "Hemorrhage",
        "gender": gender,
        "symptoms": symptoms,
        "detection": urls[0],
        "mask": urls[1]
    }

    rendered_template = render_template_string(html_template, **patient_data)



    # Image paths for hemorrhage and non-hemorrhage cases

    # Add the directory containing wkhtmltopdf to the PATH environment variable
    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

    # Use the configuration when generating PDF
    pdf = pdfkit.from_string(rendered_template, False, configuration=config, options={"enable-local-file-access": ""})

    # Now, pdfkit should be able to locate wkhtmltopdf without specifying the path explicitly

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=patient_report.pdf'

    return response

TEMP_DIR = 'temp_images'

def create_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def delete_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
def detect_hemorrhage(image_path):
    # Load the CT image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary mask
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask image to draw the detected area
    mask_image = np.zeros_like(gray)

    # Draw contours on the mask image
    for contour in contours:
        cv2.drawContours(mask_image, [contour], 0, 255, -1)

    # Apply the mask to the original image to get the masked image
    masked_image = cv2.bitwise_and(img, img, mask=mask_image)

    # Save plots as image files
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("CT Image with Hemorrhage Detection")
    plt.axis('on')
    plt.savefig('hemorrhage_detection.png')  # Save the plot as an image file
    image1_path = os.path.abspath('hemorrhage_detection.png')
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Area (Masked Image)")
    plt.axis('on')
    plt.savefig('masked_image.png')  # Save the plot as an image file
    image2_path = os.path.abspath('masked_image.png')

    # Analyze contours and calculate centroid
    for contour in contours:
        # Calculate bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate centroid (if enough mass)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw centroid
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

    # Determine and print confidential value
    if len(contours) > 0:
        confidential_value = "Confidential: Hemorrhage detected"
    else:
        confidential_value = "Confidential: No hemorrhage detected"
    print(confidential_value)

    # Display results
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("CT Image with Hemorrhage Detection")
    # plt.axis('on')

    # plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Area (Masked Image)")
    # plt.axis('on')
    plt.close()  # Close the Matplotlib figure after saving


    image_urls = [image1_path, image2_path]

    return image_urls

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request has file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # name = request.form['name']
    # age = request.form['age']
    # gender = request.form['gender']
    # symptoms = request.form['symptoms']

    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image file
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img = img.resize((150, 150))  # Assuming model input size is 150x150
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)

    # Make predictions
    # predictions = model.predict(img_array)

    # Get prediction label and probability
    # predicted_class = np.argmax(predictions[0])
    # class_label = class_labels[predicted_class]
    # probability = predictions[0][predicted_class]
    # print(class_label)
    # print(probability)

    class_label= "Hemorrhage"
    probability = 95

    # Return prediction result as JSON response
    return jsonify({
        'class_label': class_label,
        'probability': float(probability)  # Convert to float
    })

if __name__ == "__main__":
    app.run(debug=True)


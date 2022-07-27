import io
import os
import torch
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from torchvision import models
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd

test_annotation = open("test.txt").readlines()
class_id = []
label = []
for x in test_annotation:
    class_name = x.split(' ')[0].replace("_"," ")
    cleaned_name = ''.join([i for i in class_name if not i.isdigit()])
    class_id.append(cleaned_name.rstrip())
    label.append(int(x.split(' ')[1])-1)

class_names = pd.DataFrame(class_id,label).drop_duplicates().to_dict()[0]


#Function That will transform our images
def transform_image(image_bytes):
    my_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transform(image).unsqueeze(0)

#Function that makes the prediction
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return class_names[predicted_idx]

#Load the model that we trained.
model = torch.load("modelv1.pth",map_location='cpu')
model.eval()


app = Flask(__name__)


@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict', methods = ("GET","POST"))
def predict():
    if request.method == "GET":
        return 'Get Request'

    if request.method == "POST":
        image = request.files['file']
        if image.filename == '':
            return redirect(url_for("hello"))
        image_bytes = image.read()
        prediction = get_prediction(image_bytes=image_bytes)
        return render_template("result.html", prediction = prediction)

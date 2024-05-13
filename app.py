from flask import Flask, request, render_template, redirect, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug.wrappers import Response
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import json
import os
import requests
import base64
from PIL import Image, ImageDraw
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient
from io import BytesIO 

app = Flask(__name__)
CORS(app)

api_key="BIp2z6yDIqPIR1ToFHNf" ## BIp2z6yDIqPIR1ToFHNf

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="BIp2z6yDIqPIR1ToFHNf"
)

# app.wsgi_app = DispatcherMiddleware(
#     Response('Not Found', status=404),
#     {'/roboflow': app.wsgi_app}
# )

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


@app.route("/db", methods=["POST"])
@cross_origin()
def db():
        if request.method == 'POST':
            print("hi")
            if not os.path.isdir("./static/temp") and not os.path.isdir("./static/trainData"):
                #create Folder
                os.makedirs("./static/temp") 
                os.makedirs("./static/trainData")
                #get Input Image
                file=request.files["image"]
                #save Input Image in Temp Folder
                filename = file.filename
                file_path = os.path.join("./static/temp/", filename)
                file.save(file_path)
                #connect to roboflow
                rf = Roboflow(api_key=api_key)
                project = rf.workspace().project("glioblastoma-early-diagnosis-deaqt")
                model = project.version(3).model
                #save The Predict Image
                model.predict("./static/temp/"+file.filename, confidence=40).save("./static/trainData/prediction.jpg")
                #remove Old dat
                os.remove("./static/temp/"+file.filename)
            
            else:
                file=request.files["image"]
                filename = file.filename
                file_path = os.path.join("./static/temp/", filename)
                file.save(file_path)
                rf = Roboflow(api_key=api_key)
                project = rf.workspace().project("glioblastoma-early-diagnosis-deaqt")
                model = project.version(3).model
                os.remove("./static/trainData/prediction.jpg")
                model.predict("./static/temp/"+file.filename, confidence=40).save("./static/trainData/prediction.jpg")   
                os.remove("./static/temp/"+file.filename)
            return{"status":"succes"}
        else:
          return jsonify({'status': 'error', "headers": {"Access-Control-Allow-Origin"}}), 500
        

@app.route("/interfer", methods=["POST"])
@cross_origin()
def interfer():
        if request.method == 'POST':
            print("hi")
            if not os.path.isdir("./static/temp") and not os.path.isdir("./static/trainData"):
                #create Folder
                os.makedirs("./static/temp") 
                os.makedirs("./static/img/trainData")
                #get Input Image
                file=request.files["image"]
                #save Input Image in Temp Folder
                filename = file.filename
                file_path = os.path.join("./static/temp/", filename)
                file.save(file_path)
                #connect to roboflow
                image_url = "./static/temp/"+file.filename
                print(image_url)
                result = CLIENT.infer(image_url, model_id="glioblastoma-early-diagnosis-deaqt/3")
                print(result)
                image_data = requests.get(result['url']).content
                image = Image.open(BytesIO(image_data))
                image_filename = "prediction.jpg"
                image.save(image_filename)
                #remove Old dat
                os.remove("./static/temp/"+file.filename)
            
            else:
                file=request.files["image"]
                filename = file.filename
                file_path = os.path.join("./static/temp/", filename)
                file.save(file_path)
               #connect to roboflow
                image_url = "./static/temp/"+file.filename
                print(image_url)
                result = CLIENT.infer(image_url, model_id="glioblastoma-early-diagnosis-deaqt/3")
                print(result)
                # Load the image from URL
                # image_data = requests.get(image_url).content
                # image = Image.open(BytesIO(image_data))
                # Load the image from local file
                image = Image.open(image_url)
                draw = ImageDraw.Draw(image)
                
                # Draw bounding boxes on the image
                for prediction in result['predictions']:
                    x, y = prediction['x'], prediction['y']
                    width, height = prediction['width'], prediction['height']
                    draw.rectangle([x, y, x+width, y+height], outline="red", width=2)
                    draw.text((x, y), prediction['class'], fill="red")
                image_filename = "prediction.jpg"
                image.save(image_filename)
               
                os.remove("./static/temp/"+file.filename)
            return{"status":"succes"}
        else:
          return jsonify({'status': 'error', "headers": {"Access-Control-Allow-Origin"}}), 500

if __name__ == '__main__':
    # HOST = '0.0.0.0'
    # PORT = 5000
    app.run(debug=True)
    # try:
    #     app.run(HOST, PORT, debug=True)
    # except:
    #     print("hh)")
    #     app.run()

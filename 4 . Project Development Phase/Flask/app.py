from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
import os
import numpy as np

app = Flask(__name__)
model = load_model("resnet50_best.h5", compile=False)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']  
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        text = "The Eye Disease is  " + str(index[pred[0]])
        return text


if __name__ == '__main__':
    app.run(debug=True)

import flask
from flask import Flask, jsonify, render_template, url_for, request, redirect, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import os

app = Flask(__name__)

upload_folder = "static"
os.makedirs(upload_folder, exist_ok=True)

app.config["upload_folder"] = upload_folder 

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")

@app.route("/")
def home():
    return render_template("display.html")


@app.route("/segmentapi", methods = ["GET", "POST"])
def segmentbackend():
    if request.method == "POST":
        image = request.files["image"]
        image_name = secure_filename(image.filename)
        save_path = "{}/{}".format(app.config["upload_folder"], image_name)
        image.save(save_path)
        results, _ = ins.segmentImage(save_path)


        class_names = results["class_names"]
        class_ids = results["class_ids"].tolist()
        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()
        object_counts = results["object_counts"]
        masks = results["masks"]
        mask_shape = masks.shape
        masks_li = masks.tolist()

        
        return jsonify({"outputs":{"class_names": class_names, "class_ids":class_ids, "scores":scores, "object_counts":object_counts, "boxes":boxes, "mask_shape":mask_shape}, "mask_values":masks_li})

@app.route("/segmentfrontend", methods = ["GET", "POST"])
def segmentfrontend():
    if request.method == "POST":
        image = request.files["image"]
        image_name = secure_filename(image.filename)
        save_path = "{}/{}".format(app.config["upload_folder"], image_name)
        image.save(save_path)
        ins.segmentImage(save_path)

        ins.segmentImage(save_path, show_bboxes = True, output_image_name=save_path)

        return render_template("display.html", imagesource = save_path)
    

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['upload_folder'],
                               filename)

       
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000)
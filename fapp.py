from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import base64
import io
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import kurtosis, skew, entropy
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv('banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
target_count = data.auth.value_counts()
nb_to_delete = target_count[0] - target_count[1]
data = data[nb_to_delete:]

x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(x_train, y_train.values.ravel())

# Evaluate the model
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            uploaded_file = request.files['my_uploaded_file']
            if not uploaded_file.mimetype.startswith('image/'):
                return render_template("index.html", error="Invalid file type. Please upload an image file.")

            img_data = uploaded_file.read()
            img_base64 = base64.b64encode(img_data)

            # Helper functions
            def stringToImage(base64_string):
                imgdata = base64.b64decode(base64_string)
                image = Image.open(io.BytesIO(imgdata))
                return image

            def stringToEdgeImage(base64_string):
                imgdata = base64.b64decode(base64_string)
                image = Image.open(io.BytesIO(imgdata))
                img_blur = cv2.GaussianBlur(np.array(image), (3, 3), 0)
                sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
                return np.array(sobelxy)

            # Convert and preprocess the uploaded image
            opencvImage = cv2.cvtColor(np.array(stringToImage(img_base64)), cv2.COLOR_RGB2BGR)
            norm_image = np.array(opencvImage, dtype=np.float32) / 255.0

            # Compute features
            var = np.var(norm_image, axis=None)
            sk = skew(norm_image, axis=None)
            kur = kurtosis(norm_image, axis=None)
            ent = entropy(norm_image, axis=None) / 100

            # Validate computed features
            if not np.isfinite(var) or not np.isfinite(sk) or not np.isfinite(kur) or not np.isfinite(ent):
                return render_template("index.html", error="Error: Computed features contain invalid values. Please upload a valid image.")

            # Predict using the trained model
            result = clf.predict(np.array([[var, sk, kur, ent]]))
            out = "Real Currency" if result[0] == 0 else "Fake Currency"

            # Generate original and edge images
            fig = plt.figure(figsize=(3, 3))
            plt.axis('off')
            plt.imshow(stringToImage(img_base64))
            img_io = io.BytesIO()
            fig.savefig(img_io, format='svg')
            img_io.seek(0)
            original_image = img_io.getvalue().decode()

            fig2 = plt.figure(figsize=(3, 3))
            plt.axis('off')
            plt.imshow(stringToEdgeImage(img_base64))
            img_io2 = io.BytesIO()
            fig2.savefig(img_io2, format='svg')
            img_io2.seek(0)
            edge_image = img_io2.getvalue().decode()

            return render_template(
                "result.html",
                original_image=original_image,
                edge_image=edge_image,
                variance=f"{var:.2f}",
                skew=f"{sk:.2f}",
                kurtosis=f"{kur:.2f}",
                entropy=f"{ent:.2f}",
                result=out,
            )
        except Exception as e:
            print("Error during processing:", e)
            return render_template("index.html", error="An error occurred during processing. Please try again.")
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)

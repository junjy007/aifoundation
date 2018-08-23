import os
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)
upload_callbacks = None

@app.route('/')
def index():
    return render_template("upload.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    imdir = os.path.join(os.path.dirname(os.path.abspath(__name__)),
                         app.config['UPLOADED_PHOTOS_DEST'])
    if request.method == 'POST' and 'photo' in request.files:

        filename = photos.save(request.files['photo'])
        action = request.form['action']
        # TODO! RISK of exposing parent directories
        fullfname = os.path.join(imdir, filename)
        try:
            action_callback = upload_callbacks.get(action)
            action_callback(fullfname)
        except:
            print("Action failed.")
            pass
    return render_template('upload.html')

def run(debug=False):
    app.run(debug=debug)

if __name__=='__main__':
    raise EnvironmentError("Must be run as a module")

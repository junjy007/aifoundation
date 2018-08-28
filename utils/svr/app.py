from flask import Flask,render_template,request
import uuid
import base64
import numpy as np
from PIL import Image
import cv2
import io
import re
import mnist
app = Flask(__name__)
upload_callbacks = None
root='/home/nvidia/Desktop/showExample'

@app.route('/')
def hello_world():
    return render_template('index.html')

def beforeProcess(raw_imstr):
    imstr = re.sub('^data:image/.+;base64,', '', raw_imstr)  # .decode('base64')
    image_data = base64.b64decode(imstr)
    image_PIL = Image.open(io.BytesIO(image_data))
    try:
        image_np = np.array(image_PIL)[:, :, :3]
        image_cv = image_np[:, :, ::-1]
    except:
        image_cv = np.array(image_PIL)
    return image_cv

def afterProcess(output):
    cv2.imwrite('static/upload_contents/output.jpg', output)

@app.route('/upload', methods=['POST'])
def upload():
    raw_imstr = request.values['imageBase64']
    processMethod = request.values['processMethod']
    input_img = beforeProcess(raw_imstr)
    try:
        action_callback = upload_callbacks.get(action)
        output_img = action_callback(input_img)
        # Send output_img to user
    except Exception as e:
        print("Action failed.")
        print(e)
        pass

    #TODO for example mnist
    if processMethod=='MNIST':
        output=mnist.deal(input)
        outim_cv = input.copy()
        outim_cv=cv2.cvtColor(outim_cv,cv2.COLOR_GRAY2BGR)
        output=cv2.putText(outim_cv, str(output), (2, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    else:
        #this is an example, need to override
        output = input.copy()
        output = cv2.putText(output, "test", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    afterProcess(output)
    return '/static/output.jpg?uuid='+str(uuid.uuid1())

if __name__ == '__main__':
    app.run(host='0.0.0.0', ssl_context='adhoc')
#    app.run(host='0.0.0.0')

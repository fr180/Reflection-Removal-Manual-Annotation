from flask import Flask, url_for, request
import json
import jsonify
import random
import base64
from model import *

app = Flask(__name__)


@app.route('/')
def hello_world():
    return app.send_static_file('index.html')


@app.route('/ImgSave', methods=['POST'])
def img_save():
    data = json.loads(request.form['data'])
    number = random.randint(0, 9999)
    mask_array = data[0:data.rfind('|', 1)].split(',')
    mask_array = list(map(int, mask_array))
    mask_array = np.reshape(mask_array, (1,256, 256,1))

    term = np.zeros([1, 256, 256, 1], np.float32)

    term[np.where(mask_array >= 100)] = 0.8
    term = (term + 0.1) * 255.0
    mask_array = term

    cv2.imwrite("./static/result/image%04d_mask.jpg" % number, mask_array[0])

    image_array = data[data.rfind('|', 1) + 1:]
    image_array = base64.b64decode(image_array)
    file = open("./static/result/image%04d.jpg" % number, 'wb')
    file.write(image_array)
    file.close()

    gan = Pix2Pix(batch_size=1)
    input = cv2.imread("./static/result/image%04d.jpg" % number)
    input = cv2.resize(input, (256, 256))
    #input = input[0:256,0:256,:]
    input = np.reshape(input, (1, 256, 256, 3))
    mask = cv2.imread("./static/result/image%04d_mask.jpg" % number)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("./static/result/image%04d_mask.jpg" % number, mask)
    mask = np.resize(mask, (1, 256, 256, 1))
    result = gan.predict_single_image(input, mask)
    cv2.imwrite("./static/result/image%04d_result.jpg" % number, result)
    print(number)
    return "%04d" % number


if __name__ == '__main__':
    app.debug = True
    app.run("0.0.0.0")

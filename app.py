import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import time
from threading import Thread
import queue
from flask import jsonify
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import time

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
messages = []
inputImage = None
# device = torch.device("cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
vis_processors = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/reset_data', methods=['POST'])
def reset_data():
    print('rest--------------------------------------')
    messages.clear()
    return "Data reset successfully", 200

@app.route('/is_first_question', methods=['GET'])
def is_first_question():
    if len(messages) > 0:
        return jsonify(is_first=False)
    else:
        return jsonify(is_first=True)

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def generate_answer(prompt):
    # Replace this with your answer generation logic
    answer = model.generate({"image": inputImage, "prompt": prompt})
    return answer[0]

@app.route('/send_message', methods=['POST'])
def send_message():
    global inputImage
    context = []
    for i in range(0, len(messages), 2):
        question = messages[i]['message']
        answer = messages[i+1]['message']
        context.append((question, answer))
    print('context', context)

    print('send_message_in...........')
    username = request.form.get('username')
    message = request.form.get('message')
    image = request.files.get('image')
    print('message: ', message)
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_url = url_for('uploaded_file', filename=filename)
        messages.append({'username': username, 'message': message, 'image_url': image_url})
        raw_image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).convert("RGB")
        width, height = raw_image.size
        print('inputImage_shape_width_height', width, height)
        inputImage = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    elif message:
        messages.append({'username': username, 'message': message, 'image_url': None})

    template = "Question: {} Answer: {}."
    prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + message + " Answer:"
    print(prompt)
    # answer_queue.put('new answer')
    start_time = time.time()
    answer = generate_answer(prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("generate_answer elapsed: {:.2f} 秒".format(elapsed_time))
    messages.append({'username': 'A', 'message': answer, 'image_url': None})

    # return redirect(url_for('index'))
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    app.run(debug=False, host='0.0.0.0', port=3000)

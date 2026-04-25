from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from process_video import process_video

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = f'processed_{filename}'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # process video (draw boxes and labels)
    try:
        process_video(input_path, output_path)
    except Exception as e:
        # show a simple error page with message
        return render_template('error.html', error_message=str(e))

    return redirect(url_for('result', filename=output_filename))


@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)


@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

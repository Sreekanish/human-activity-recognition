from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import threading
import numpy as np
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Define model and other constants used in predict_class_video
# Defining variables
FRAME_HEIGHT, FRAME_WIDTH = 64, 64 # shape of the image
SEQUENCE_LENGTH =10 # Frames feeded to the model as a single sequence
CLASSES_LIST = ['Archery', 'Basketball', 'Biking', 'Bowling', 'CricketBowling','CricketShot', 'Diving', 'Fencing', 'GolfSwing', 'HighJump',
'HulaHoop', 'PlayingFlute', 'PlayingGuitar', 'PlayingViolin', 'PlayingCello','Skijet', 'SkyDiving', 'Skiing', 'Surfing', 'Typing','TableTennisShot', 'LongJump'] # Example class list, replace with your actual classes
model = load_model('lrcn_model2__loss_')  

def convert_avi_to_mp4(input_filepath, output_filepath):
    video = VideoFileClip(input_filepath)
    video.write_videofile(output_filepath)
    video.close()# You need to load your model here

def predict_class_video(video_file_path: str, sequence_length: int=20):
    # Reading the video file
    video_reader = cv2.VideoCapture(video_file_path)
    
    # Extracting frames at certain interval for max sequence length.
    frames_list = []
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(int(total_frames/sequence_length), 1)
    
    # Looping and extracting
    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames)
        success, frame = video_reader.read()
        if not success:
            break
        resize_frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
        norm_frame = resize_frame/255.
        frames_list.append(norm_frame.astype('float32'))
    
    # predicting using the LRCN model 
    pred_prob = model.predict(np.expand_dims(frames_list, axis=0))[0]
    pred_label = np.argmax(pred_prob)
    pred_class = CLASSES_LIST[pred_label]
    # printing the result
    print(f'[INFO] Action Predicted: {pred_class}')
    print(f'[INFO] Prediction Probabilities: {pred_prob[pred_label]:.2f}\n')
    video_reader.release()
    return pred_prob, pred_label, pred_class
@app.route('/')
def open():
    return render_template('open.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global filepath,filename,pred_class
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    # print("filewnaaaaaaaaaaaa",file)
    sequence_length=SEQUENCE_LENGTH

    if file.filename == '':
        return redirect(request.url)

    #filename = file.filename
    filename = os.path.basename(file.filename)
  
   
    #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    if filename.endswith('.avi'):
    # Convert AVI to MP4
        mp4_filepath = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + '.mp4')
        convert_avi_to_mp4(filepath, mp4_filepath)
        os.remove(filepath)  # Remove the original AVI file
        filename = os.path.basename(mp4_filepath)
        filepath = mp4_filepath
    # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        pred_prob, pred_label, pred_class = predict_class_video(filepath, sequence_length)
    #print("qqqqqqqqqqqqqq",pred_prob, pred_label, pred_class)


    # Process the uploaded video in a separate thread
    video_thread = threading.Thread(target=predict_class_video, args=(filepath,sequence_length))
    video_thread.start()
    print("bbbbbbbbbbbbbbbbbbbbbbbb",video_thread)

    return redirect(url_for('display_video'))

@app.route('/display_video')
def display_video():
    #print("fffffffffffffffffffffffffff",filepath)
    print("00000000000000000000000000",filename)
    print("qqqqqqqqqqqqqq", pred_class)
    
    return render_template('video.html',abc=filename,ab=pred_class)


if __name__ == '__main__':
    app.run(debug=True)

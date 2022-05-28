import os
import soundfile as sf
import streamlit as st
from scipy.io import wavfile
from code import record,save
import matplotlib.pyplot as plt
import numpy as np
import librosa, librosa.display
from code import *

from utils.splitTrainDev import predict
#preprocessing images
import tensorflow as tf
from tensorflow.keras import layers, models,preprocessing,regularizers,callbacks
from matplotlib import pyplot as plt

def setup_empty_model():
    # model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((4, 4), strides=4))
    model.add(layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.MaxPooling2D((1, 3), strides=(1,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='linear',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.8))
    model.add(layers.experimental.RandomFourierFeatures(output_dim=10))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy',
                        tf.keras.metrics.TrueNegatives(),
                        tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.FalseNegatives(),
                        tf.keras.metrics.FalsePositives()])
    return model

def setup_model():
    model = setup_empty_model()
    model.load_weights('models/cnn_svm_weights.h5')
    return model

def predict(img):
    model = setup_model()
    img = preprocessing.image.load_img(img, target_size=(512, 512))
    img = preprocessing.image.img_to_array(img)
    xinput = np.expand_dims(img, axis=0)
    return model.predict(xinput)

st.set_page_config(page_title="Audio", layout="wide")

st.sidebar.title('Dementia Detection')
options = ['🎤Record','💾Upload','🔨Process']
choice = st.sidebar.radio('Select an option',options)
if choice == options[0]:
    with st.form('Record your audio'):
        st.title('🎤Record')
        st.write('Record a short audio clip of 25 seconds')
        filename = st.text_input("Enter Person name (no extensions)")
        rec_btn = st.form_submit_button('Record for 25 seconds')
    if rec_btn and filename:
        with st.spinner("recording"):
            filepath = 'sounds/'+filename.split('.')[0]+"_recording.wav"
            st.info(f"will save audio to {filepath} ")
            path = record(25,WAVE_OUTPUT_FILENAME=filepath,)
            st.success(f"voice recorded as {path}")
    else:
        st.info('click to start after entering filename')
if choice == options[1]:
    with st.form('upload your audio'):
        st.title("💾Upload")
        audio = st.file_uploader("select an audio file",type=['wav'])
        record_audio = st.form_submit_button('Upload')
    if record_audio:
        if audio is not None:
            audio_file = save(audio,f'sounds/{audio.name}')
            st.success(f"upload uploaded to sounds/{audio.name}")
        else:
            st.error("please select an audio file")

if choice == options[2]:
    with st.form('Process'):
        st.title("🔨Process")
        st.write('Process sound file(Speech Data) and generate MFCC and STFT')
        soundfiles= os.listdir('sounds')
        file = st.selectbox('Select a sound file',soundfiles)
        
        process_audio = st.form_submit_button('Process')
    if process_audio:
        if file is not None:
            st.audio(f'sounds/{file}')
            audio_file = 'sounds/'+file
            st.info(f"processing {audio_file}")
            y, sr = librosa.load(audio_file)
            
            fig,ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr)
            ax.set_title('Waveform')
            st.pyplot(fig)
            c1,c2 = st.columns(2)
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            fig2,ax = plt.subplots(figsize=(12, 10))
            librosa.display.specshow(mfccs, x_axis='time', y_axis='mel',ax=ax)
            ax.set_title('MFCC')
            c1.pyplot(fig2)
            stft = librosa.stft(y)
            fig3,ax = plt.subplots(figsize=(12, 10))
            librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), x_axis='time', y_axis='log',ax=ax)
            ax.set_title('STFT')
            stft_file = 'datasets/'+file.split('.')[0]+"_stft.png"
            fig3.savefig(stft_file)
            c2.pyplot(fig3)
            st.success(f"stft saved to {stft_file}") 
            prediction = predict(stft_file)  
            if prediction == 0:
                st.markdown(f"# Voice shows sign of depression 😖 in analysis")
            else:
                st.markdown(f"# Voice does not show sign of depression 😀 in analyis")   

        else:
            st.error("please select a file")
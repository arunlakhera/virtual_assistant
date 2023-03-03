import pyttsx3
import speech_recognition as sr
from playsound import playsound
import random
import datetime
import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from modules import commands_answers, load_agenda
sns.set()

commands = commands_answers.commands
answers = commands_answers.answers

# print(commands)
# print(answers)

my_name = "Prince"

chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'


def search(sentence):
    wb.get(chrome_path).open("https://www.google.com/search?q=" + sentence)


MODEL_TYPES = ['EMOTION']


def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('models/speech_emotion_recognition.hdf5')
        model_dict = list(['calm', 'happy', 'fear', 'nervous', 'neutral', 'disgust', 'surprise', 'sad'])
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE

# print(load_model_by_name('EMOTION')[0].summary())


model_type = 'EMOTION'
loaded_model = load_model_by_name(model_type)


def predict_sound(AUDIO, SAMPLE_RATE, plot = True):
    results = []
    wav_data, sample_rate = librosa.load(AUDIO, sr = SAMPLE_RATE)
    # print(wav_data.shape)
    # print(sample_rate)
    # print(wav_data)
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end=True, pad_value=0)

    for i, data in enumerate(splitted_audio_data.numpy()):
        # print('Audio Split:', i)
        # print(data.shape)
        # print(data)
        mfccs_features = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        # print(mfccs_features.shape)
        # print(mfccs_features)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        # print(mfccs_scaled_features.shape)
        mfccs_scaled_features = mfccs_scaled_features[:, :, np.newaxis]
        # print(mfccs_scaled_features.shape)
        predictions = loaded_model[0].predict(mfccs_scaled_features)
        print(predictions)
        # print(predictions.sum())
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            plt.show()

        predictions = predictions.argmax(axis = 1)
        # print(predictions)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)
        # print(results)

        results_str = 'PART ' + str(i) + ':' + str(predictions).upper()
        # print(results_str)

    count_results = [[results.count(x), x] for x in set(results)]
    # print(count_results)

    # print(max(count_results))
    return max(count_results)

# playsound('./sad.wav')
# predict_sound('sad.wav', loaded_model[2], plot=False)

def play_music_youtube(emotion):
    play = False

    if emotion == 'sad' or emotion == 'fear':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=pPH2zrX4-iQ')
        play = True
    if emotion == 'nervous' or emotion == 'surprise':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=fkqvUvAwPTs')
        play = True
    return play

# play_music_youtube('sad')
# play_music_youtube('nervous')

# emotion = predict_sound('sad.wav', loaded_model[2], plot=False)
# print(emotion)
# play_music_youtube(emotion[1])


def speak(text):

    engine = pyttsx3.init()
    engine.setProperty('rate', 125) # number of words per second
    engine.setProperty('volume', 1) # min 0 max 1
    engine.say(text)
    engine.runAndWait()

# speak("To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags")

def listen_microphone():
    microphone = sr.Recognizer()

    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source, duration=0.8)
        print('Listening: ')
        audio = microphone.listen(source)

        with open('recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())

    try:
        # https://pypi.org/project/SpeechRecognition/
        sentence = microphone.recognize_google(audio, language='en-US')
        print('You said:' + sentence)

    except sr.UnknownValueError:
        sentence = ''
        print('Not Understood')
    return sentence


listen_microphone()

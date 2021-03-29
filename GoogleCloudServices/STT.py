import os
from os import walk

dirname = os.path.dirname(__file__)+"\\AudioFiles\\"
filename = os.path.join(dirname,'Test1.wav') 
keypath = os.path.join("C:","Users","HP","Documents","SeniorDesign","apt-theme-305701-ed17d438daf1.json")
import speech_recognition as sr

# print(keypath)

r=sr.Recognizer()
with sr.AudioFile(filename) as source:
    try:
        audio = r.record(source)
        print(r.recognize_google(audio_data=audio,language="en-US",key=keypath))
    except sr.UnknownValueError as e:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
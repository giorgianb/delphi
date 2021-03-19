import os
from os import walk

dirname = os.path.dirname(__file__)+"\\AudioFiles\\Test1_TwoMinuteSegments\\"
_, _, filenames = next(walk('./AudioFiles/Test1_TwoMinuteSegments/'))

keypath = os.path.join("C:","Users","HP","Documents","SeniorDesign","apt-theme-305701-ed17d438daf1.json")

import wave

import speech_recognition as sr

for fn in filenames:
    n = os.path.join(dirname,fn)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(n) as source:
            audio = r.record(source)
            print("\n"+fn)
            # print(r.recognize_google_cloud(audio, language="en-US", credentials_json=keypath))
            print(r.recognize_google(audio, language="en-US", key="ed17d438daf1ecdf14765ebe77225959f96c8f9a"))
    except sr.UnknownValueError as e:
        print("Unknown Value Error: "+str(e))
    except (sr.RequestError) as e:
        print("Request Error: "+str(e))
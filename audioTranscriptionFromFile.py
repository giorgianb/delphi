import os
from os import walk

dirname = os.path.dirname(__file__)+"\\AudioFiles\\Test1_TwoMinuteSegments\\"
# filename = os.path.join(dirname,'./AudioFiles/Test1_TwoMinuteSegments/Test1000.wav')

_, _, filenames = next(walk('./AudioFiles/Test1_TwoMinuteSegments/'))

import wave

import speech_recognition as sr

for fn in filenames:
    n = os.path.join(dirname,fn)
    # print(fn)
    # print(n)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(n) as source:
            audio = r.record(source)
            print("\n"+fn)
            print(r.recognize_google(audio, language="en-US"))
    except sr.UnknownValueError as e:
        print(e)

# print(filename)
# d = 120
# o = 316

# with sr.AudioFile(filename) as source:
    # audio = r.record(source, duration=d, offset=o)
    # print(r.recognize_google(audio))

    # for i in range(3):
    #     print("the offset is "+str(o)+" the duration is "+str(d)+".\n")
    #     audio = r.record(source, duration=d, offset=o)
    #     print(r.recognize_google(audio))
    #     o+=d
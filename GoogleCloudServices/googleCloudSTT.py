from google.cloud import speech
import os
from os import walk

client = speech.SpeechClient()

dirname = os.path.dirname(__file__)+"\\AudioFiles\\" # THIS PATH DOESN'T WORK BECAUSE IT MUST BE A GOOGLE CLOUD STORAGE URI 
filename = os.path.join(dirname,'Test1.wav') # THIS CODE GIVES THE ERROR PATH IS AN INVALID GCS path

audio = speech.RecognitionAudio(uri=filename)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
)

response = client.recognize(config=config, audio=audio)

for result in response.results:
    print("Transcript: {}".format(result.alternatives[0].transcript))
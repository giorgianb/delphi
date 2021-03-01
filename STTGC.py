# WORKING SCRIPT FOR FILES UNDER 1 MINUTE 

from google.cloud import speech
import io
import os
from os import walk

dirname = os.path.dirname(__file__)+"\\AudioFiles\\Test3_UnderOneMinuteSegments\\"
_, _, filenames = next(walk('./AudioFiles/Test3_UnderOneMinuteSegments/'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='C:\\Users\\HP\\Documents\\SeniorDesign\\apt-theme-305701-ed17d438daf1.json'

client = speech.SpeechClient()

for fn in filenames:
    n = os.path.join(dirname,fn)
    with io.open(n, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=32000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print("\n"+fn)
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
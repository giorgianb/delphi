use ffmpeg to convert .mp4 to .wav
      # ffmpeg -i Test1.mp4 output_audio.raw
      # ffmpeg -i Test1.mp4 output_audio.wav
      # ffmpeg -i Test1.mp4 Test1.m4a
      # ffmpeg -i Test1.m4a output_audio.wav
use ffmpeg to split .wav into 2 minute audio segments
      # ffmpeg -i Test1.wav -f segment -segment_time 120 -c copy Test1%03d.wav
      # ffmpeg -i Test1.wav -f segment -segment_time 900 -c copy mTest1%03d.wav
use audioTranscriptionFromFile.py to convert from .wav to 180s chuncks of text
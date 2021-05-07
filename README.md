# Delphi
Automatically Generate a Table of Contents for Lecture Videos

## Usage
The usage of Delphi is divided into several stages. The first is the training stage, in which Delphi learns how to create table of contents for lecture videos for a course. The first stage involves training Delphi to recognize spoken technical terms in the lecture video, and training Delphi how to assign subjects to sentences.

### Phase 1: Delphi Training
Training Delphi involves two portions, which can be done independently.
#### DeepSpeech Training
Due to the inadequacies of major transcription services (Google, Microsoft, IBM), we use *DeepSpeech* in order to generate transcriptions of lecture videos. These transcriptions are then fed into a model which assigns a subject to each sentence in the transcriptions, and divides the transcription into segments based on topics.

###### Obtaining DeepSpeech
Deepspeech can be obtained using the following code:

    # Create and activate a virtualenv
    virtualenv -p python3 $HOME/tmp/deepspeech-venv/
    source $HOME/tmp/deepspeech-venv/bin/activate

    # Install DeepSpeech
    pip3 install deepspeech

    # Download pre-trained English model files
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
    git clone --branch v0.9.3 https://github.com/mozilla/DeepSpeech
    
###### Training DeepSpeech

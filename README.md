# Delphi
Automatically Generate a Table of Contents for Lecture Videos

## Usage
The usage of Delphi is divided into several stages. The first is the training stage, in which Delphi learns how to create table of contents for lecture videos for a course. The first stage involves training Delphi to recognize spoken technical terms in the lecture video, and training Delphi how to assign subjects to sentences.

### Phase 1: Delphi Training
Training Delphi involves two portions, which can be done independently.
#### DeepSpeech Training
Due to the inadequacies of major transcription services (Google, Microsoft, IBM), we use *DeepSpeech* in order to generate transcriptions of lecture videos. These transcriptions are then fed into a model which assigns a subject to each sentence in the transcriptions, and divides the transcription into segments based on topics.

###### Obtaining DeepSpeech
We assume that you have [conda](https://www.anaconda.com/products/individual) installed. Deepspeech can be obtained using the following commands:

    # Create and activate a conda enviroment
    conda create -n deepspeech python tensorflow-gpu==1.15 sox cudatoolkit

    # Install DeepSpeech
    pip install deepspeech

    # Download pre-trained English model files
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
    
    # Download the Deepspeech Code for Training
    git clone --branch v0.9.3 https://github.com/mozilla/DeepSpeech
    
    
    # Install necessary dependenices
    cd DeepSpeech
    pip install --upgrade pip==20.2.2 wheel==0.34.2 setuptools==49.6.0
    pip install --upgrade -e .
    
    make Dockerfile.train
    
    # Create a directory for training and move necessary files/folders into it.
    mkdir training
    cd training
    mv ../../deepspeech-0.9.3-models.pbmm .
    mv ../../deepspeech-0.9.3-models.scorer .
    
    
###### Training DeepSpeech
The next step is to train Deepspeech in order to recognize the technical terms that are spoken in the lecture video, as these were not included in *DeepSpeech*'s training vocabulary. To do this, the user must provide at least one audio WAV file that contains all of the technical terms that are said during all of the lecture. The audio file MUST have a `16kHz` samplign rate. It is recommended that the technical terms be said multiple times, in some random order, for best results. A human-generated transcription of each of the WAV files is also necessary. Once these are obtained, create the CSV file with the following fields, named `train.csv`:

        wav_filepath, wav_filesize, transcript
 
 Ensure that the above header is not omitted from the CSV file, as DeepSpeech will look for the CSV header and produce an error if it is not there. The transcription must contain only English alphabetical characters - no punctuation or numerals are allowed. Each row in the CSV file, aside from the header, will contain the `wav_filepath`, `wave_filesize`, and `transcript` of the audio WAV file with the technical terms said during the lectures.
 
 Once `train.csv` is created in the `training/` folder, run the following command to train `DeepSpeech`:
 
        cd ../
        python3 DeepSpeech.py --train_cudnn --n_hidden 2048 --checkpoint_dir training/ --epochs 3 --train_files training/train.csv --learning_rate 0.0001
 

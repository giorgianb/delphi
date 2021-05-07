# Delphi
Automatically Generate a Table of Contents for Lecture Videos.

![ESE_441_Poster](https://user-images.githubusercontent.com/11166439/117481306-cdbdef80-af30-11eb-9cf0-0e38b2e60d88.png)


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
    
    # Create directories for training and move necessary files/folders into it.
    mkdir pretrained
    mv../deepspeech-0.9.3-models.pbmm pretrained/
    mv../deepspeech-0.9.3-models.scorer pretrained/
    
    mkdir training
    cd training
    
    
###### Training DeepSpeech
The next step is to train Deepspeech in order to recognize the technical terms that are spoken in the lecture video, as these were not included in *DeepSpeech*'s training vocabulary. To do this, the user must provide at least one audio WAV file that contains all of the technical terms that are said during all of the lecture. The audio file MUST have a `16kHz` samplign rate. It is recommended that the technical terms be said multiple times, in some random order, for best results. A human-generated transcription of each of the WAV files is also necessary. Once these are obtained, create the CSV file with the following fields, named `train.csv`:

        wav_filepath, wav_filesize, transcript
 
 Ensure that the above header is not omitted from the CSV file, as DeepSpeech will look for the CSV header and produce an error if it is not there. The transcription must contain only English alphabetical characters - no punctuation or numerals are allowed. Each row in the CSV file, aside from the header, will contain the `wav_filepath`, `wave_filesize`, and `transcript` of the audio WAV file with the technical terms said during the lectures.
 
 Once `train.csv` is created in the `training/` folder, run the following command to train `DeepSpeech`:
 
        cd ../
        python3 DeepSpeech.py --train_cudnn --n_hidden 2048 --load_checkpoint_dir pretrained/ -save_checkpoint_dir training/ --epochs 3 --train_files training/train.csv --learning_rate 0.0001
 
 Once training, is completed rename the latest `.pbmm` file and `.scorer` file in the `training` directory to `delphi.pbmm` and `delphi.scorer`. These are the files that will be used to generate transcriptions.
 
 #### SECTOR Training
Besides from training DeepSpeech to recognize spoken technical terms, *SECTOR* must be trained in order to perform topic classification and segmentation.

##### Generating the Training Data
The training data to train *SECTOR* is obtained from the PDF of a textbook. For best result, the textbook should be the textbook used in order to teach the course. The textbook PDF must also have a table of contents with *PDF Bookmarks*. A textbook has this sort of table of contents when there are entries in the PDF document viewer that can be clicked in order to automatically arrive at the desired chapter/section. Once a suitable PDF is obtained, it can be used in order to generate the training data for SECTOR using *Apollo*.

First, download Delphi and install the necessary dependencies for *Apollo*:

        git clone https://github.com/giorgianb/delphi
        cd delphi
        conda create -n apollo python ipython
        pip install stanza nltk pyenchant
        python -c 'import nltk; nltk.download("stopwords"); nltk.download("wordnet"); import stanza; stanza.download("en")'
        
Now, use Apollo to generate the training data. The argument `course_textbook.pdf` is the path to the textbook used for the course, and `data.csv` is the path to the CSV file that will hold the training data. This path is important as it will be used later.

       bin/apollo.sh course_textbook.pdf data.csv
       



 

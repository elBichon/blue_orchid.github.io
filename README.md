# Blue orchid
###### Blue orchid is a script that when given a language and a subject will extract data from wikipedia, using the section titles as labels and the sentences between two labels as training data. Then, a LSTM classifier will be trained and saved to be re-used later
###### Edit, as LSTM were not very efficient as th.ere was not enough data, I decided to switch to random forest

## Motivation: 
The project is to easily create and serve text classifier in any language on any given subject using the wide open knowledge repository available in wikipedia. Others objectives where also to get a better understanding of LSTM and arg-parse

## Tech/Frameworks used:
- [keras](https://keras.io/)
- spacy
- pandas
- beautifull soup
- wikipedia api

## Project Structure:
- blue_orchid.py
- utils.py

## Features:
- [x] Extracting the data from wikipedia and save them as a dataset
- [x] Defining the right machie learning model to train the model
- [x] Saving the model
- [x] Serving the model
- [x] Create an executable that uses argparse, to ask for training or using and then asks for the language and subject or model and sentence to classify
- [] Adding exception handling
- [] Creating a docker container


## Workflow:

*Training
1. Given as input a language and a subject
2. The ad hoc dataset will be created
3. The LSTM network will be trained on those data
4. The model will be saved as the_subject_language.h5

*using the model
1. Given as input a language, a subject and a string to classify
2. The ad hoc model will be selected and loaded
3. The string will go through the network and classify the string 

## How to use:

```
git status
git add
git commit
```

import streamlit as st
import pandas as pd
from PIL import Image
from operator import itemgetter
from itertools import groupby
import re
import nltk
from nltk import word_tokenize,pos_tag
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SpacyTokenizer
import stanza

@st.experimental_singleton
def load_models():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    spacy_model = spacy.load('en_core_web_sm')
    flair_model=SequenceTagger.load('ner')
    stanza.download('en')
    stanza_model = stanza.Pipeline('en', download_method=None)
    return [spacy_model,flair_model,stanza_model]

models=load_models()
spacy_nlp=models[0]
tagger=models[1]
stanza_nlp=models[2]

def main():
    st.title("Welcome to my anonymization tool!")
    menu=["About","Evaluation of Packages","Visualize the Process","Anonymize Manual Input","Anonymize txt File","Anonymize CSV File"]
    choice=st.sidebar.radio("Section:",menu)

    if choice == "About":
        exec(open("about.py").read())

    elif choice == "Evaluation of Packages":  
        exec(open("evaluation_of_packages.py").read())

    elif choice == "Visualize the Process":
        
        exec(open("visualize_the_process.py").read())

    elif choice == "Anonymize Manual Input":
        exec(open("anonymize_manual_input.py").read())

    elif choice == "Anonymize txt File":
        exec(open("anonymize_txt.py").read())

    else:
        exec(open("anonymize_csv.py").read())

if __name__ == '__main__':
    main()
    
if st.button("Please clear all once you are done with the app, to prevent memory limit issues on Streamlit"):
    # Clears all singleton caches:
    st.experimental_singleton.clear()

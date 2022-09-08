import streamlit as st
import pandas as pd
from PIL import Image

st.title("Evaluation of Packages")

data_ev=st.container()
with data_ev:
    data_ev.header("Data used for Evaluation")
    data_ev.write("I have done an evaluation of the packages using a modified version of the WikiNeural dataset obtained from [here](https://github.com/Babelscape/wikineural). Only name masking is considered in this section.")
    citation=data_ev.expander("Citation for WikiNeural Dataset")
    citation.markdown('''Tedeschi, S., Maiorca, V., Campolungo, N., Cecconi, F., & Navigli, R. (2021). 
    WikiNEuRal: Combined Neural and Knowledge-based Silver Data Creation for Multilingual NER.
    In Findings of the Association for Computational Linguistics: EMNLP 2021 (pp. 2521–2533). 
    Association for Computational Linguistics.
    ''')
    data=pd.read_csv("modified_data.csv")
    data_ev.write(data.head())
    data_ev.caption("Double click on the cell to view the full sentence stored in the cell")
    data_ev.caption("A total of 1000 sentences was used in this evaluation")
    data_ev.subheader("Data Visualization")
    dist_image=Image.open('word_distribution.jpg')
    data_ev.image(dist_image,caption='The 1000 sentences used for the final evaluation can contain varying number of words, ranging from 3 words to 112 words. Most of the sentences contain around 13 to 23 words',width=600)
    name_image=Image.open('name_image.jpg')
    data_ev.image(name_image,caption='Most of the sentences used in the final evaluation contain only 1 personal name(Identified by [Name] in the expected output). There is 1 sentence with 16 personal names.',width=600)

ev_results=st.container()
with ev_results:
    ev_results.header("Evaluation Results")
    ev_results.subheader("Recall and Precision")
    ev_results.write("Recall and Precision were calculated for all packages on a sentence level and an overall level.")
    ev_results.latex(r''' Recall = {True\ Positive \over True Positive + False\ Negative } 
              = {Number\ of\ correct\ [Name]\ tag\ by\ package \over Number\ of\ [Name]\ tag\ in\ original\ sentence} ''')
    ev_results.latex(r'''Precision = {True\ Positive \over True\ Positive + False\ Positive }
                 = {Number\ of\ correct\ [Name]\ tag\ by\ package \over Number\ of\ [Name]\ tag\ by\ package} ''')
    ev_results.latex(r''' Overall\ Recall = {Total\ number\ of\ correct\ [Name]\ tag\ by\ package\ over\ 1000\ sentences \over Total\ number\ of\ [Name]\ tag\ in\ original\ 1000\ sentence} ''')
    ev_results.latex(r'''Overall\ Precision = {Total\ number\ of\ correct\ [Name]\ tag\ by\ package\ over\ 1000\ sentences \over Total\ number\ of\ [Name]\ tag\ by\ package\ over\ 1000\ sentences}''')
    package_list=["","NLTK","spaCy","flair","stanza","union","intersection"]
    values=[["Precision (Sentence Level)","0.838","0.869","0.956","0.902","0.834","0.884"],["Precision (Overall Level)","0.815","0.869","0.944","0.895","0.804","0.888"],
            ["Recall (Sentence Level)","0.682","0.616","0.852","0.814","0.847","0.520"],["Recall (Overall Level)","0.704","0.639","0.867","0.838","0.870","0.543"]]
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    ev_results.markdown(hide_table_row_index, unsafe_allow_html=True)
    rp=pd.DataFrame(values)
    rp.columns=package_list
    ev_results.table(rp)
    ev_results.subheader("Time Taken and Peak Memory Block")
    ev_results.write("The time package and tracemalloc package were used to compute the time taken and peak memory block used by in each scenario respectively")
    with_package,without_package=ev_results.columns(2)
    header_for_time_memory=["Package Involved","Time Taken (s)","Peak Memory Block (MB)"]
    with with_package:
        with_package.subheader("Importing of packages considered")
        value_results=[["NLTK","52","116"],["spaCy","17","102"],["flair","724","1068"],["stanza","605","253"],["union","1425","1080"],["intersection","1414","1080"]]
        time_memory=pd.DataFrame(value_results)
        time_memory.columns=header_for_time_memory
        with_package.table(time_memory)
        
    with without_package:
        without_package.subheader("Importing of packages not considered")
        value_results1=[["NLTK","50","72"],["spaCy","12","6"],["flair","711","6"],["stanza","622","6"],["union","1397","82"],["intersection","1391","82"]]
        time_memory1=pd.DataFrame(value_results1)
        time_memory1.columns=header_for_time_memory
        without_package.table(time_memory1)
        
    ev_results.caption('''Some observations:
- It can be observed that spaCy and NLTK are the faster packages, but at the expense of recall and precision. Although flair and stanza gave a higher recall and precision, they took a much longer time, at least 30 times longer than that of spaCy, and at least 10 times longer than that of NLTK.

- The union and intersection options took a much longer time than the individual packages since they involve all packages. However, the intersection option did not do very well, which is not surprising since it is more restrictive.

- It appears that time taken is not significantly affected by the importing of packages but the peak memory block is. This suggests that the packages and models contribute significantly to the peak memory block.

Note:

- Exact values may vary. Current results are obtained on a MacBook Air M1 Processor and rounded up

- A while loop is used under NLTK to obtain the start and end character index of the names identified. For spaCy, flair and stanza, these information could be obtained directly from the entities that the package has labelled.''')

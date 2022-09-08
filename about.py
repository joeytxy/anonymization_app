import streamlit as st
import pandas as pd
from PIL import Image

st.title("Anonymization Tool")
st.markdown("This is an anonymization tool which utilises NER packages (flair, NLTK,spaCy and stanza) to mask personal names (default). Other information such as NRIC, phone number etc can also be masked by giving corresponding input.")

packages=st.container()
with packages:
    packages.header("Package Information")

    nltk=packages.container()
    with nltk:
        nltk_info=nltk.expander("Information about NLTK")
        nltk_info.subheader("Citation")
        nltk_info.write("Steven Bird, Ewan Klein, and Edward Loper (2009). Natural Language Processing with Python. O’Reilly Media Inc. ")
        nltk_info.subheader("Documentation")
        nltk_info.write("For more information regarding the NLTK package, its documentation can be found [here](https://www.nltk.org/)")
        
    spacy=packages.container()
    with spacy:
        spacy_info=spacy.expander("Information about spaCy")
        spacy_info.subheader("Citation")
        spacy_info.write('''message: "If you use spaCy, please cite it as below."

      authors:
      
    - family-names: "Honnibal"
  
      given-names: "Matthew"
    
    - family-names: "Montani"

      given-names: "Ines"

    - family-names: "Van Landeghem"

      given-names: "Sofie"

    - family-names: "Boyd"

      given-names: "Adriane"
        
    title: "spaCy: Industrial-strength Natural Language Processing in Python"
      
    doi: "10.5281/zenodo.1212303"
      
    year: 2020
    ''')
        spacy_info.subheader("Documentation")
        spacy_info.write("For more information regarding the spaCy package, you can either visit their [web page](https://spacy.io/) or their [GitHub page](https://github.com/explosion/spaCy)")

    flair=packages.container()
    with flair:
        flair_info=flair.expander("Information about flair")
        flair_info.subheader("Citation")
        flair_info.write('''@inproceedings{akbik2019flair,
      title={{FLAIR}: An easy-to-use framework for state-of-the-art {NLP}},
      author={Akbik, Alan and Bergmann, Tanja and Blythe, Duncan and Rasul, Kashif and Schweter, Stefan and Vollgraf, Roland},
      booktitle={{NAACL} 2019, 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)},
      pages={54--59},
      year={2019}
    }''')
        flair_info.subheader("Documentation")
        flair_info.write("For more information regarding the flair package, its GitHub page can be found [here](https://github.com/flairNLP/flair)")

    stanza=packages.container()
    with stanza:
        stanza_info=stanza.expander("Information about stanza")
        stanza_info.subheader("Citation")
        stanza_info.write('''Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. 
    Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. 
    In Association for Computational Linguistics (ACL) System Demonstrations. 2020. ''')
        stanza_info.subheader("Documentation")
        stanza_info.write("Article can be found [here](https://arxiv.org/abs/2003.07082)")
        stanza_info.write("To download its paper in pdf format directly, click [here](https://nlp.stanford.edu/pubs/qi2020stanza.pdf)")
        stanza_info.write("More information on the framework or citations can be found [here](https://stanfordnlp.github.io/stanza/index.html)")

    packages.subheader("Union/Intersection of Packages")
    union_image=Image.open('./union_intersection.png')
    packages.image(union_image,caption='Union vs Intersection of Packages')
    union_explanation=packages.expander("Is 'union' always the best option?")
    union_explanation.write("The idea of 'union' may be appealing as it seems that we are taking all 'correct' names from various packages. However, there are situations where 'union' may not perform better than an individual package. An example is illustrated below:")
    eg1_image=Image.open('./example1.png')
    union_explanation.image(eg1_image,caption='When does \'union\' do worse?')
    union_explanation.markdown("It can be observed that flair provides us with the correct output since <font color='red'>'s</font> should not be considered as part of a person's name. However if we were to take an union of the stanza package and the flair package, <font color='red'>Peter Jackson 's</font> is masked due to stanza's tagging, giving us an incorrect input.", unsafe_allow_html=True)
    
model=st.container()
with model:    
    model.header("Model Methodology")
    model.subheader("What is string slicing and chracter index?")
    model.markdown("sentence=\"<font color='blue'>**Mary Lee ate pasta. She met Anna at the restaurant.**</font> \"", unsafe_allow_html=True) 
    sentence="Mary Lee ate pasta. She met Anna at the restaurant."
    character_list=[]
    index_list=[]
    for i in range(len(sentence)):
        character_list.append(sentence[i])
        index_list.append(i)
    mytable=pd.DataFrame(character_list).T
    mytable.columns=index_list
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    model.markdown(hide_table_row_index, unsafe_allow_html=True)
    model.table(mytable)
    model.write("sentence[0:8] will give us 'Mary Lee'. Note that the stop index 8 is not included") 
    model.subheader("Model Flowchart")
    idea_image=Image.open('./model_idea.png')
    model.image(idea_image,caption='Model Idea')
    model_explanation=model.expander("Click here to view explanation for usage of character index")
    model_explanation.markdown('''While it is possible to use the entity text identified as PERSON by the packages directly, character index was used in this implementation to provide users with an union/intersection option. Using character index ensures that the union/intersection function is applied on the same word.

Suppose we have this sentence: <font color='blue'>**"Kim went to her office today. She had a meeting with Mr Kim."** </font>

Package A tags "Kim" as PERSON. However, there is a possibility that A recognises the first "Kim" as a PERSON and not the second occurence. Let us assume that that is the case.

**Package A: <font color='red'>Kim</font> went to her office today. She had a meeting with Mr Kim**.

Suppose we have another package B that tags both "Kim" as PERSON.

**Package B: <font color='red'>Kim</font> went to her office today. She had a meeting with Mr <font color='red'>Kim</font>**.

Without the use of character index to determine, the intersection of both packages would have given the following output:

**'<font color='red'>[Name]</font> went to her office today. She had a meeting with Mr <font color='red'>[Name]</font>.'**

While this is a correct output, it does not fit the definition of intersection.

By using character index, we can recognise that A recognises character index 0 to 3 as a person name, and not character index 56 to 59. The intersection betweeen [0,1,2,3] and [0,1,2,3,35,57,58,59] would have given [0,1,2,3]. Masking the word at sentence[0:3] would have given us:

**'<font color='red'>[Name]</font> went to her office today. She had a meeting with Mr Kim.'**

which fits our definition of intersection.

Word index was not used as different packages may tokenise the sentence differently, resulting in the same word having a different index.''',unsafe_allow_html=True)

    model.subheader("Masking of other personal details")
    model.write("Regular expression is used to mask other personal details. The case is ignored (specified in a separate argument).Click on the relevant sections to view more about the respective details")
    nric=model.container()
    with nric:
        nric_info=nric.expander("NRIC")
        nric_info.markdown('''Regular expression : <font color='green'> **r\"([sftg]\d{7}[a-z])\"** </font>

Replaced with : <font color='green'> **[NRIC]** </font>

This matches with any text that starts with <font color='green'> **S/F/T/G** </font>, followed by <font color='green'> **7 numeric digits** </font>, and ends with <font color='green'> **any alphabet** </font>. This is case insensitive.

Example : <font color='green'> **S1234567A** </font>
''' ,unsafe_allow_html=True)

    caseno=model.container()
    with caseno:
        case_info=caseno.expander("Case Number")
        case_info.markdown('''Regular expression : <font color='gold'> **r"(\d{10}[A-z])"** </font>

Replaced with : <font color='gold'> **[CASENO]** </font>

The regular expression matches to any text starts with <font color='gold'> **10 consecutive digits**</font> followed by either <font color='gold'> **an alphabet or the following symbols: [ \ ] ^ _ `**</font> This is case insensitive.

Example : <font color='gold'> **1234567890A** </font>
''' ,unsafe_allow_html=True)

    phoneno=model.container()
    with phoneno:
        phone_info=phoneno.expander("Phone Number")
        phone_info.markdown('''Regular expression : <font color='DeepSkyBlue'> **r"(\d{8})"** </font>

Replaced with : <font color='DeepSkyBlue'> **[PHONE]** </font>

The regular expression matches any <font color='DeepSkyBlue'> **8 consecutive digits**</font>.

Example : <font color='DeepSkyBlue '> **91008100** </font>
''' ,unsafe_allow_html=True)

    idx=model.container()
    with idx:
        idx_info=idx.expander("ID")
        idx_info.markdown('''Regular expression : <font color='hotpink'> **r"([a-z]\d{4}[a-z])"** </font> or <font color='hotpink'> **r"(\d{5}[a-z])"** </font>

Replaced with : <font color='hotpink'> **[ID]** </font>

The first regular expression matches to any text that starts with <font color='hotpink'> **an alphabet**</font>, followed by <font color='hotpink'> **4 consecutive digits**</font> and <font color='hotpink'> **an alphabet**</font>.

The second regular expression matches any text that starts with <font color='hotpink'> **5 consecutive digits**</font>, followed by <font color='hotpink'> **any alphabet**</font>.

This is case insensitive.

Example : <font color='hotpink '> **a1234Z** </font> or <font color='hotpink '> **12345A** </font>
''' ,unsafe_allow_html=True)

    date=model.container()
    with date:
        date_info=idx.expander("Date")
        date_info.markdown('''Regular expression : <font color='MediumAquaMarine'> **r"(\d{1,2}.\d{1,2}.\d{2,4})"** </font> or <font color='MediumAquaMarine'> **r"(\d{1,2}.(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?).\d{2,4})"** </font>

Replaced with : <font color='MediumAquaMarine'> **[DATE]** </font>

The first regular expression matches any text that starts with <font color='MediumAquaMarine'> **either 1 or 2 consecutive digits**</font>, followed by <font color='MediumAquaMarine'> **any character except newline**</font>, followed by <font color='MediumAquaMarine'> **either 1 or 2 consecutive digits**</font>, followed by <font color='MediumAquaMarine'> **any character except newline**</font>, followed by <font color='MediumAquaMarine'> **either 2 or 4 consecutive digits.**</font>

The second regular expression matches any text that starts with <font color='MediumAquaMarine'> **either 1 or 2 consecutive digits**</font>, followed by <font color='MediumAquaMarine'> **any character except newline**</font>, followed by <font color='MediumAquaMarine'> **a month which can be of an abbreviated form (eg Jan) or the full form (January)**</font>, followed by <font color='MediumAquaMarine'> **any character except newline**</font>, followed by <font color='MediumAquaMarine'> **either 2 or 4 consecutive digits.**</font>

This is case insensitive.

Example : <font color='MediumAquaMarine'> **1/1/22**</font> or <font color='MediumAquaMarine'> **21-12-2022**</font> or <font color='MediumAquaMarine'> **05/04/2012**</font> or <font color='MediumAquaMarine'> **1 January 2012**</font>
or <font color='MediumAquaMarine'> **05 aug 22**</font>
''' ,unsafe_allow_html=True)

    atime=model.container()
    with atime:
        atime_info=atime.expander("Admission Time")
        atime_info.markdown('''Regular expression : <font color='rosybrown'> **r"(admission Time.\s\d+.\d+)"** </font>

Replaced with : <font color='rosybrown'> **Admission Time: [Time]** </font>

The regular expression matches any text that starts with the phrase "<font color='rosybrown'> **admission Time**</font>", followed by <font color='rosybrown'> **any character except newline**</font>, followed by <font color='rosybrown'> **any white space character**</font>, followed by <font color='rosybrown'> **one of more digit**</font>, followed by <font color='rosybrown'> **any character except newline**</font>, followed by <font color='rosybrown'> **one or more digit**</font>. This is case insensitive.

Example : <font color='rosybrown'> **admission time: 2:45**</font> or <font color='rosybrown'> **Admission time: 12.30**</font>  

''' ,unsafe_allow_html=True)

    wardno=model.container()
    with wardno:
        ward_info=wardno.expander("Ward Number")
        ward_info.markdown('''Regular expression : <font color='darkgoldenrod'> **r"(ward.\w+\s[a-zA-z0-9]+)"** </font>

Replaced with : <font color='darkgoldenrod'> **Ward:[WardNo]** </font>

The regular expression matches any text that starts with the phrase "<font color='darkgoldenrod'> **ward**</font>", followed by <font color='darkgoldenrod'> **any character except newline**</font>, followed by <font color='darkgoldenrod'> **at least one occurence of a word character i.e letters, alphanumeric, digits and underscore**</font>, followed by <font color='darkgoldenrod'> **any whitespace characters**</font>, followed by <font color='darkgoldenrod'> **at least one occurance of alphabets/digits/the following symbols: [ \ ] ^ _ `**</font>. This is case insensitive.

Example : <font color='darkgoldenrod'> **ward:type b1**</font> or <font color='darkgoldenrod'> **ward type A**</font>  

''' ,unsafe_allow_html=True)

    bedno=model.container()
    with bedno:
        bed_info=bedno.expander("Bed Number")
        bed_info.markdown('''Regular expression : <font color='darkorchid'> **r"(bed.\s[a-z0-9]+)"** </font>

Replaced with : <font color='darkorchid'> **Bed:[BedNo]** </font>

The regular expression matches any text that starts with the phrase "<font color='darkorchid'> **bed**</font>", followed by <font color='darkorchid'> **any character except newline**</font>, followed by <font color='darkorchid'> **any whitespace chracter**</font>, followed by <font color='darkorchid'> **at least one occurence of alphabet/digit.**</font> This is case insensitive.

Example : <font color='darkorchid'> **bed: a12**</font> or <font color='darkorchid'> **BED: 10**</font>  

''' ,unsafe_allow_html=True)


    pclass=model.container()
    with pclass:
        pclass_info=pclass.expander("Patient Class")
        pclass_info.markdown('''Regular expression : <font color='lawngreen'> **r"(patient class.\s\w+\s[A-Z])"** </font>

Replaced with : <font color='lawngreen'> **Patient Class:[Class]** </font>

The regular expression matches any text that start with the phrase "<font color='lawngreen'> **patient class**</font>", followed by <font color='lawngreen'> **any character except newline**</font>, followed by <font color='lawngreen'> **any whitespace character**</font>, followed by <font color='lawngreen'> **at least one occurence of a word character i.e letters, alphanumeric, digits and underscore**</font>, followed by <font color='lawngreen'> **any whitespace character**</font>, followed by <font color='lawngreen'> **any alphabet**</font>. This is case insensitive.

Example : <font color='lawngreen'> **patient class: Private A**</font>

''' ,unsafe_allow_html=True)

    model.write("If there are other expressions you would like to mask, you may also give a pattern for the tool")
    add_expression=model.container()
    with add_expression:
        otherexp_info=add_expression.expander("How to include additional expressions")
        otherexp_info.write("Suppose you would like to mask age and dates that do not include years. Please provide the relevant information in order. An example is shown in the screenshot below:")
        otherexp_image=Image.open('./other_expression.png')
        otherexp_info.image(otherexp_image,caption='Example of how to include additional expressions',width=300)


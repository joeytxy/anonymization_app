import streamlit as st
import pandas as pd
        
def anonymized_text_color(user_input,package=['stanza'],union_intersection=None,additional_details=None,additional_expression=None):    
    colored_text=user_input
    final_return=user_input

    # to obtain full list of index for eg [0,8]->[0,1,2,3,4,5,6,7,8]
    def index_list(list1):   
        final_list=[]
        for i in list1:
            for j in range(i[0],i[1]+1):
                final_list.append(j)
        return final_list

    #to obtain range for each set of consecutive numbers after union/intersection for eg [0,1,2,3,4,5] -> [0,5]
    def range_lists(list1):   
        output=[]
        for k, g in groupby(enumerate(list1), lambda x: x[0]-x[1]):
            group=list(map(itemgetter(1), g))
            output.append([group[0],group[-1]])
        return output

    #to obtain identified names for eg user_input[0:5]
    def name_list(list1): 
        final_names=[]
        for i in list1:
            final_names.append(user_input[i[0]:i[1]])
        final_names=sorted(final_names,key=len,reverse=True)
        return final_names


    accumulated=[]

    if 'nltk' in package:
        if user_input.strip()=="":
            accumulated.append([])
        else: 
            df=pd.DataFrame()

            #obtain word and corresponding tag
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(user_input))):
                if hasattr(chunk,'label'):
                    for c in chunk:
                        data={'word':[c[0]],'label':[chunk.label()]}
                        tmp = pd.DataFrame(data)
                        df=pd.concat([df,tmp])

                else:
                    data={'word':[chunk[0]],'label':[chunk[1]]}
                    tmp = pd.DataFrame(data)
                    df=pd.concat([df,tmp])
            counter=0
            list_of_indices=[]
            df['word']=df['word'].str.replace("\`\`","\"",regex=True)
            df['word']=df['word'].str.replace("\'\'","\"",regex=True)
            #search for word's start index to end index in user_input
            for i in df['word']:
                while counter<len(user_input):
                    if i==user_input[counter:counter+len(i)]:
                        list_of_indices.append([counter,counter+len(i)])
                        counter=counter+len(i)
                        break
                    else:
                        counter=counter+1
            df['index']=list_of_indices

            #to obtain person name 
            df=df[df['label']=="PERSON"] #obtain a df (example row: Anna  PERSON  [0,4])
            nltk_index_list=[]

            #append each range (start char index, end char index) of name to list
            #combine if the words are consecutive (to eliminate problem of identifying first name and last name as two names)
            for i in df['index']:
                if len(nltk_index_list)>0 and i[0]==nltk_index_list[-1][1]+1:
                    nltk_index_list[-1][1]=i[1]
                else:
                    nltk_index_list.append(i)

            #if user chooses to just use nltk to anonymize text then use each range directly
            if len(package)==1:
                accumulated.append(nltk_index_list)

            #if need to union/intersect, obtain full list of character index      
            else:
                nltk_index=index_list(nltk_index_list)
                accumulated.append(nltk_index)

    if 'spacy' in package:
        spacy_doc = spacy_nlp(user_input)
        spacy_index_list=[]
        #append range (start char index,end char index) of person name identified to list
        for ent in spacy_doc.ents:
            if ent.label_=="PERSON":
                if len(spacy_index_list)>0 and ent.start_char==spacy_index_list[-1][1]+1:
                    spacy_index_list[-1][1]=ent.end_char
                else:
                    spacy_index_list.append([ent.start_char,ent.end_char])

        #if user chooses to just use spacy to anonymize text then use each range directly 
        if len(package)==1:
            accumulated.append(spacy_index_list)

        #if need to union/intersect, obtain full list of character index
        else:
            spacy_index=index_list(spacy_index_list)
            accumulated.append(spacy_index)

    if 'flair' in package:
        text=Sentence(user_input,use_tokenizer=SpacyTokenizer(spacy_nlp))
        tagger.predict(text)
        flair_index_list=[]
        #append range (start char index,end char index) of person name identified to list
        for entity in text.get_spans('ner'):
            if entity.get_label('ner').value=="PER":
                if len(flair_index_list)>0 and entity.start_position==flair_index_list[-1][1]+1:
                    flair_index_list[-1][1]=entity.end_position
                else:
                    flair_index_list.append([entity.start_position,entity.end_position])

        #if user chooses to just use flair to anonymize text then use each range directly 
        if len(package)==1:
            accumulated.append(flair_index_list)

        #if need to union/intersect, obtain full list of character index
        else:
            flair_index=index_list(flair_index_list)
            accumulated.append(flair_index)

    if 'stanza' in package:
        stanza_doc=stanza_nlp(user_input)
        stanza_index_list=[]
        #append range (start char index,end char index) of person name identified to list
        for i in range(0,len(stanza_doc.sentences)):
            for entity in stanza_doc.entities:
                if entity.type=="PERSON":
                    if len(stanza_index_list)>0 and entity.start_char==stanza_index_list[-1][1]+1:
                        stanza_index_list[-1][1]=entity.end_char
                    else:
                        stanza_index_list.append([entity.start_char,entity.end_char])

        #if user chooses to just use stanza to anonymize text then use each range directly 
        if len(package)==1:
            accumulated.append(stanza_index_list)

        #if need to union/intersect, obtain full list of character index
        else:
            stanza_index=index_list(stanza_index_list)
            accumulated.append(stanza_index)

    if union_intersection!=None and union_intersection.lower()=='union':
        #obtain union of lists given by relevant packages
        def union(list1):
            return list(set().union(*list1))

        #sort list to check for consecutive numbers
        sorted_list=union(accumulated)
        sorted_list.sort()
        union_list=range_lists(sorted_list)

        #obtain name list
        name_to_mask=name_list(union_list)

    elif union_intersection!=None and union_intersection.lower()=='intersection':
        #obtain intersection of lists given by relevant packages
        def intersect(list1):
            return list(set.intersection(*map(set, list1)))

        #sort list to check for consecutive numbers
        sorted_list=intersect(accumulated)
        sorted_list.sort()
        intersection_list=range_lists(sorted_list)

        #obtain name list
        name_to_mask=name_list(intersection_list)
    else:   #case where only one package is used
        if len(accumulated[0])!=0: 
            name_to_mask=name_list(accumulated[0])
        else:
            #case where no personal names were identified 
            name_to_mask=[]

    #sort name list to ensure full name is masked first before masking instances where only first name is used 
    name_to_mask=sorted(list(set(name_to_mask)), key=len,reverse=True)
    for i in name_to_mask:
        colored_text=colored_text.replace(i,''.join(["<font color='red'> **",i,"** </font>"]))
        final_return=final_return.replace(i,"[Name]")
        final_return=final_return.replace("[Name]","<font color='red'> **[Name]** </font>")

    #this is for cases where part of this name is tagged as name by the package
    #to ensure that the full name is colored
    for i in re.findall(r"([*]{2}\s</font>\s[A-Za-z\s]+[*]{2}\s</font>)",colored_text):
        colored_text=colored_text.replace(i,i[10:])
    colored_text=colored_text.replace("<font color='red'> **<font color='red'>","<font color='red'>")

    #mask additional details if requested     
    if additional_details!=None: 
        if 1 in additional_details:
            for i in re.findall(r"([sftg]\d{7}[a-z])",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='green'> **",i,"** </font>"]))
            final_return = re.sub(r"([sftg]\d{7}[a-z])", "[NRIC]", final_return,flags=re.IGNORECASE) 
            final_return=final_return.replace("[NRIC]","<font color='green'> **[NRIC]** </font>")
        if 2 in additional_details:
            for i in re.findall(r"(\d{10}[A-z])",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='gold'> **",i,"** </font>"]))
            final_return = re.sub(r"(\d{10}[A-z])", "[CASENO]", final_return, flags=re.IGNORECASE)
            final_return=final_return.replace("[CASENO]","<font color='gold'> **[CASENO]** </font>")
        if 3 in additional_details:
            for i in re.findall(r"(\d{8})",final_return):
                colored_text=colored_text.replace(i,''.join(["<font color='DeepSkyBlue'> **",i,"** </font>"]))
            final_return = re.sub(r"(\d{8})", "[PHONE]", final_return)
            final_return=final_return.replace("[PHONE]","<font color='DeepSkyBlue'> **[PHONE]** </font>")
        if 4 in additional_details:
            for i in re.findall(r"([a-z]\d{4}[a-z])",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='hotpink'> **",i,"** </font>"]))
            for i in re.findall(r"(\d{5}[a-z])",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='hotpink'> **",i,"** </font>"]))
            final_return = re.sub(r"([a-z]\d{4}[a-z])", "[ID]", final_return, flags=re.IGNORECASE)
            final_return = re.sub(r"(\d{5}[a-z])", "[ID]", final_return, flags=re.IGNORECASE)
            final_return=final_return.replace("[ID]","<font color='hotpink'> **[ID]** </font>")
        if 5 in additional_details:
            for i in re.findall(r"(\d{1,2}.\d{1,2}.\d{2,4})",final_return):
                colored_text=colored_text.replace(i,''.join(["<font color='MediumAquaMarine'> **",i,"** </font>"]))
            for i in re.findall(r"(\d{1,2}.(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?).\d{2,4})",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='MediumAquaMarine'> **",i,"** </font>"]))
            final_return = re.sub(r"(\d{1,2}.\d{1,2}.\d{2,4})", "[DATE]", final_return)
            final_return = re.sub(r"(\d{1,2}.(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?).\d{2,4})", "[DATE]",final_return,flags=re.IGNORECASE)
            final_return=final_return.replace("[DATE]","<font color='MediumAquaMarine'> **[DATE]** </font>")
        if 6 in additional_details:
            for i in re.findall(r"(admission Time.\s\d+.\d+)",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='rosybrown'> **",i,"** </font>"]))
            final_return = re.sub(r"(admission Time.\s\d+.\d+)", "Admission Time: [Time]", final_return, flags=re.IGNORECASE)
            final_return=final_return.replace("Admission Time: [Time]","<font color='rosybrown'> **Admission Time: [Time]** </font>")
        if 7 in additional_details:
            for i in re.findall(r"(ward.\w+\s[a-zA-z0-9]+)",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='darkgoldenrod'> **",i,"** </font>"]))
            final_return = re.sub(r"(ward.\w+\s[a-zA-z0-9]+)", "Ward:[WardNo]", final_return, flags=re.IGNORECASE)
            final_return=final_return.replace("Ward:[WardNo]","<font color='darkgoldenrod'> **Ward:[WardNo]** </font>")
        if 8 in additional_details:
            for i in re.findall(r"(bed.\s[a-z0-9]+)",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='darkorchid'> **",i,"** </font>"]))
            final_return = re.sub(r"(bed.\s[a-z0-9]+)", "Bed:[BedNo]", final_return, flags=re.IGNORECASE)
            final_return=final_return.replace("Bed:[BedNo]","<font color='darkorchid'> **Bed:[BedNo]** </font>")
        if 9 in additional_details:
            for i in re.findall(r"(patient class.\s\w+\s[A-Z])",final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(i,''.join(["<font color='lawngreen'> **",i,"** </font>"]))
            final_return = re.sub(r"(patient class.\s\w+\s[A-Z])", "Patient Class:[Class]", final_return, flags=re.IGNORECASE)
            final_return=final_return.replace("Patient Class:[Class]","<font color='lawngreen'> **Patient Class:[Class]** </font>")

    if additional_expression!=None:
        for i in additional_expression:
            for j in re.findall(i[0],final_return,flags=re.IGNORECASE):
                colored_text=colored_text.replace(j,''.join(["<font color='navy'> **",j,"** </font>"]))
            final_return = re.sub(i[0],i[1],final_return,flags=re.IGNORECASE)
            final_return = final_return.replace(i[1],''.join(["<font color='navy'> **",i[1],"** </font>"]))
    return [colored_text,final_return]
st.caption("A run button will appear at the end of the page when all required details are given")
st.title("Lets Visualize!")
option=st.radio("Would you like to see a color-coded output for a single manual input or a txt file?",("Single Manual Input","txt file"))
input1=None
if option=="Single Manual Input":
    input1=st.text_area("Write something to run")
else:
    file_received=st.file_uploader("What text file would you like to anonymize?", type=['txt'])
    if file_received is not None:
        input1=""
        for i in file_received:
            input1+=i.decode("utf-8")
        
package_choice,other_detail,other_expression=st.columns(3)
package_choice.subheader("Step 1")
package=package_choice.multiselect("Select the package(s) you would like to use. You may also select more than one package",['nltk','spacy','flair','stanza'])
union_intersection=None
if len(package)>1:
    union_intersection=package_choice.radio("Would you like to take an union or an intersection?",('union','intersection'))

other_detail.subheader("Step 2 (Optional)")
other_detail.write("What other personal details would you like to mask?")
additional_details=[]
nric=other_detail.checkbox("NRIC")
if nric:
    additional_details.append(1)
caseno=other_detail.checkbox("Case Number")
if caseno:
    additional_details.append(2)
phoneno=other_detail.checkbox("Phone Number")
if phoneno:
    additional_details.append(3)
idno=other_detail.checkbox("ID")
if idno:
    additional_details.append(4)
date=other_detail.checkbox("Date")
if date:
    additional_details.append(5)
atime=other_detail.checkbox("Admission Time")
if atime:
    additional_details.append(6)
wardno=other_detail.checkbox("Ward Number")
if wardno:
    additional_details.append(7)
bedno=other_detail.checkbox("Bed Number")
if bedno:
    additional_details.append(8)
pclass=other_detail.checkbox("Patient Class")
if pclass:
    additional_details.append(9)
if len(additional_details)==0:
    additional_details=None

other_expression.subheader("Step 3 (Optional)")
num=other_expression.number_input("How many extra regular expressions would you like to mask?",min_value=0,max_value=10)
additional_expression=[]
for i in range(0,int(num)):
    regular_expression=other_expression.text_input("What is the regular expression?",key=str(i)+"_reg")
    replacement=other_expression.text_input("What would you like to replace it with?",key=str(i)+"_rep")
    additional_expression.append([regular_expression,replacement])
empty=0
if len(additional_expression)!=0:
    for i in additional_expression:
        if i[0]=="":
            empty+=1
        if i[1]=="":
            empty+=1
if len(additional_expression)==0:
    additional_expression=None
if input1 is not None:
    if package!=[]:
        if empty==0:
            anonymize_now=st.button("Run")
            if anonymize_now:
                results=anonymized_text_color(input1,package,union_intersection,additional_details,additional_expression)
                original,anonymized=st.columns(2)
                original.subheader("Original Text")
                original.markdown(results[0],unsafe_allow_html=True)
                anonymized.subheader("Anonymized Text")
                anonymized.markdown(results[1],unsafe_allow_html=True)
                st.snow()

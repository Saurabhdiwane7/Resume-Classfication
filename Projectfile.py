import pandas as pd
import re,nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer,RegexpStemmer,WordNetLemmatizer,wordnet
from nltk import word_tokenize
from nltk.probability import FreqDist
import plotly.express as  px
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import pickle
import docx2txt
from wordcloud import WordCloud, ImageColorGenerator
import pdfplumber
from nltk import RegexpTokenizer
import streamlit as st

model=pickle.load(open('rf.pkl','rb'))
vector = pickle.load(open('tf.pkl','rb'))

nltk.download('wordnet')
nltk.download('stopwords')




#below function add all resumes in empty list 
resume =[]   # empty list

def display(doc_file):
    if doc_file.type ==  "application/vnd.openxmlformats-officedocument.wordprocessingml.document":   #thic code checking if file format is docx.
        resume.append(docx2txt.process(doc_file)) # extract docx to txt and appending  it to resume list
    else :
        with pdfplumber.open(doc_file) as pdf: #for pdf format
            pages = pdf.pages[0]                   # extracting pdf pages 
            resume.append(pages.extract_text())    # convert pages to txt and append to resume 
    return resume
    


#below function cleaned the data using regular expressions
def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('<HTML\S+>',"")  # removal of html tags
    sentence = re.sub('<.*?>',"", sentence)    # html tags
    sentence =re.sub('http\S+', '', sentence)    #http links
    rem_num = re.sub('[0-9]+','',sentence)
    tokenizer = RegexpTokenizer(r'\w+')  #tokenize the sentences
    tokens = tokenizer.tokenize(rem_num)
    filtered_words =[ w for w in tokens if len(w)>2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return ' '.join(lemma_words)
    



 #below function found most common words and their frequency    
def most_common(cleaned,i):               # gives text,frequency values 
    tokenizer = RegexpTokenizer(r'\w+')   #imported only text to tokenize
    words = tokenizer.tokenize(cleaned)   ##tokenize the text data
    mostcommon = FreqDist(cleaned.split()).most_common(i)    #splited the data and checked most common words
    return mostcommon


#below function create wordclouds
def wordcloud(mostcommon):
    wordcloud=WordCloud(width=1000, height=600, background_color='black').generate(str(mostcommon))
    a=px.imshow(wordcloud)
    st.plotly_chart(a)
    wc = WordCloud(background_color='Black',width=1000,height=600).generate(str(most_common))
    a= px.imshow(wc)
    st.plotly_chart(a)
  


    
# below function create bar chart of words and its counts  
def display_words(mostcommon_small):
    x,y= zip(*mostcommon_small)
    chart= pd.DataFrame({x :'key',y:'values'})
    fig= px.bar(chart,x=chart['keys'],y=chart['values'],height=700,width=700)
    st.plotly_chart(fig)
    
def main():
    st.title('RESUME CLASSIFICATION')
    upload_file = st.file_uploader('Upload Files Here ',
                                type= ['docx','pdf'],accept_multiple_files=True)
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1].upper(),
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed= display(doc_file)

                cleaned=preprocess(display(doc_file))
                predicted= model.predict(vector.transform([cleaned]))

                string='The Resume category  is belongs to '+predicted[0]
                st.header(string)

                st.subheader('WORDCLOUD')
                wordcloud(most_common(cleaned, 100))

                st.header('Frequency of 20 Most Common Words')
                display_words(most_common(cleaned, 30))

if __name__ == '__main__':
    main()
    

       




    
   
    
    
    


    
                          
                    
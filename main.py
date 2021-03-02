import fitz
import spacy
from nlp import nlp as nlp
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher
from collections import Counter
import Levenshtein as lev
import matplotlib.pyplot as plt
from wordcloud import WordCloud

Spnlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(Spnlp.vocab)

def read_pdf_and_format(pdf_document):
    content = ""
    cnt = 0
    doc = fitz.open(pdf_document)
    while cnt < doc.pageCount:
        page = doc.loadPage(cnt)
        content = content + page.getText("text")
        cnt+=1
    return content
def read_txt_and_format(txt_document):
    with open(txt_document) as job:
        text = job.read()  
    return text
def clean_text(corpus):
    cleaned_text = ""
    for i in corpus:
        cleaned_text = cleaned_text + i.lower().replace("'", "")
    return cleaned_text


def main(resume, job_description):
    resume_text = read_pdf_and_format(resume)
    print('\n\n', summarize(resume_text, ratio=0.2))    # ------------------------------------ ==> Resume Summary
    print('\n\n', summarize(clean_text(job_description), ratio=0.2))    # -------------------------------- ==> Job Description Summary
    text_list = [resume_text, job_description]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2)
    print('\n\n', matchPercentage)    # ------------------------------------ ==> Match Percentage
    tokensJob = nlp().tokenize(job_description)
    job = Counter(tokensJob).most_common(20)
    data = []
    for r in job:
        rec = {}
        rec['word'] = r[0]
        rec['from'] = 'job'
        rec['freq'] = r[1]
        data.append(rec)
    print('\n\n\n')
    responseJson = {'resumeText': resume_text, 'jobDescription': job_description, 'keywords': [], 'match': matchPercentage}
    for i in data:
        if i['freq'] > 2:
            responseJson['keywords'].append({'skill': i['word'], 'freq': i['freq'], 'keySkill': True})
        else:
            responseJson['keywords'].append({'skill': i['word'], 'freq': i['freq'], 'keySkill': False})
    print(responseJson)    # ------------------------------------ ==> Keywords Data

if __name__ == '__main__':
    main("input.pdf", read_txt_and_format("job_description.txt"))
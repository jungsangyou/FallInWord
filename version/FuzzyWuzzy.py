from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from operator import itemgetter
from konlpy.tag import Twitter
#from pykospacing import spacing
import re
import numpy as np
import json, random

#fuzz.ratio("this is a test", "this is a test!")
#fuzz.partial_ratio("this is a test", "this is a test!")
#fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
#fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
#fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")


class FuzzyWuzzy() :

    def __init__(self, file=None, texts=[], confidence=0.6) :
        self.texts = []
        self.t = Twitter()
        self.confidence = confidence * 100
        self.clusterings = []
        self.id = 0

        for uid, text in enumerate(texts):
            #tmp_text = self.filtering([spacing(line.strip('\n'))])
            tmp_text = self.filtering([text])
            self.texts.append({
                'text' : tmp_text,
                'original' : text
            })



    # stopwords를 제거하는 부분
    def filtering(self, str_list):
        str_list = list(map(lambda x: re.sub('[\?\.]', '', x), str_list))
        str_list = list(map(lambda x: re.sub('어떻게 해야 하나요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('어떻게 되는 것인가요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('어떻게 되나요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('왜 그런가요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('라고 나와요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('되나요', '', x), str_list))
        #str_list = [' '.join(re.split(' ', str_list[0])[0:-1])]
        #print(str_list)
        str_pos = self.t.pos(str_list[0], stem=True)
        #print(str_pos)
        #str_filt = np.array(str_pos)[np.where(np.in1d(list(map(itemgetter(1), str_pos)), ['Noun', 'Verb', 'Adjective', 'Alpha', 'Foreign', 'Number']))[0]]
        str_filt = np.array(str_pos)[np.where(np.in1d(list(map(itemgetter(1), str_pos)), ['Noun', 'Alpha', 'Foreign', 'Number']))[0]]

        if len(str_filt) > 1 and str_filt[-1][1] != 'Noun' :
            str_filt = str_filt[0:-1]

        str_final = [' '.join(list(map(itemgetter(0), str_filt)))]
        stop_words = ['가능하다','어떻다','어떤', '무슨', '알다', '말', '하다', '되다', '궁금하다', '가능하다', '시', '수', '인가요', '있다', '하나요', '해야하나요', '좋다', '해', '요', '한', '가요', '대해']
        split_str_list = list(map(lambda x: re.split(' ', x), str_final))

        filtered_word = list(map(lambda x: ' '.join(list(np.array(x)[np.logical_not(np.in1d(x, stop_words))])), split_str_list))
        return filtered_word[0]
        #return jaso_split(filtered_word[0])


    def run(self):

        for i, text_ob in enumerate(self.texts) :
            if i%100 == 0 :
                if self.confidence > 51 :
                    self.confidence = self.confidence - 1

            if i == 0 :
                self.create_clustering(text_ob['original'], text_ob['text'])
            else :
                self.ratio(text_ob)

        print(json.dumps(self.clusterings))

    def ratio(self, text_ob):
        text = text_ob['text']
        original = text_ob['original']
        max_category = 0
        max_ratio = 0
        max_len = 0
        random.shuffle(self.clusterings)
        for ob in self.clusterings :
            this_ratio = fuzz.token_set_ratio(text, ob['totalText'])
            if max_ratio < this_ratio :
                max_category = ob['category']
                max_ratio = this_ratio

        if max_ratio > self.confidence :
            self.add_clustering(max_category, original, text)
        else :
            self.create_clustering(original, text)

    def create_clustering(self, original, text):
        tmp_totalTexts = list(set(re.split(' ', text)))
        text = ' '.join(list(set(tmp_totalTexts)))
        cluster = {
            "category" : self.id,
            "texts" : [original],
            "totalText" : text,
        }
        self.clusterings.append(cluster)
        self.id = self.id + 1


    def add_clustering(self, category, original, text):
        for cluster in self.clusterings :
            if cluster['category'] == category :
                cluster['texts'].append(original)
                #cluster['totalText'] = cluster['totalText'] + ' ' + text

                tmp_totalTexts = re.split(' ', cluster['totalText'])
                if len(tmp_totalTexts) < 13 :
                    #print(self.t.pos(text, stem=True))
                    cluster['totalText'] = cluster['totalText'] + ' ' + text
                    tmp_totalTexts = list(set(re.split(' ', cluster['totalText'])))

                cluster['totalText'] = ' '.join(tmp_totalTexts)

                return

    def model_to_clustering(self, json=None):

        return

    def clustering_to_model(self):
        return

if __name__ == "__main__":
    inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample2.txt"
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing.txt"
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_bmt.txt"

    texts = []
    for uid, line in enumerate(open(inputfile)):
        texts.append(line.strip('\n'))
    random.shuffle(texts)

    FuzzyWuzzy(texts=texts, confidence=0.7).run()





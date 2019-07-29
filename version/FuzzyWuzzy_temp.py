from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from operator import itemgetter
from konlpy.tag import Twitter
from collections import Counter
#from pykospacing import spacing
import re
import numpy as np
import json, random

#fuzz.ratio("this is a test", "this is a test!")
#fuzz.partial_ratio("this is a test", "this is a test!")
#fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
#fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
#fuzz.token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")


############
# v2 -> v3
############
# 1. 로직 변경
#  -> 전반 적인 로직 변경, 이유는 명사를 기준으로 하면 분류가 정확도가 높지만, 동사 기준으로 분류 된 샘플은 분류 정확도가 낮음
# 2. 로직 변경 설명(beta)
#  -> 아래와 같은 순으로 로직 구성
#    (1) 명사기준 + 컨피던스값 높게 (0.8 -> 0.7), batch_size 2 이상
#    (2) 동사까지 넣고 빡세게 묶고 (0.8 이상), batch_size 10 이상


############

class FuzzyWuzzy() :

    def __init__(self, clusterings=[], texts=[], confidence=0.6, batch_size=0) :
        self.before_texts = texts
        self.texts = []
        self.batch_size = batch_size
        self.t = Twitter()
        self.confidence = confidence * 100
        self.clusterings = []
        self.id = 0
        self.texts_len = 13
        if len(clusterings) > 0 :
            self.convert_clustering(clusterings=clusterings)

    # stopwords를 제거하는 부분
    def filtering(self, str_list=None, noun=False):
        str_list = list(map(lambda x: re.sub('[\?\.]', '', x), str_list))
        str_list = list(map(lambda x: re.sub('어떻게 해야 하나요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('어떻게 되는 것인가요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('어떻게 되나요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('왜 그런가요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('라고 나와요', '', x), str_list))
        str_list = list(map(lambda x: re.sub('되나요', '', x), str_list))

        str_pos = self.t.pos(str_list[0], stem=True)
        stop_pos = ['Noun', 'Alpha', 'Foreign', 'Number']
        if not(noun) :
            stop_pos.append('Verb')
            stop_pos.append('Adjective')

        str_filt = np.array(str_pos)[np.where(np.in1d(list(map(itemgetter(1), str_pos)), stop_pos))[0]]

        if noun and len(str_filt) > 1 and str_filt[-1][1] != 'Noun' :
            str_filt = str_filt[0:-1]

        str_final = [' '.join(list(map(itemgetter(0), str_filt)))]
        stop_words = ['어떤', '무슨', '알다', '말', '하다', '되다', '궁금하다', '가능하다', '시', '수', '인가요', '있다', '하나요', '해야하나요', '좋다', '해', '요', '한', '가요', '대해']
        split_str_list = list(map(lambda x: re.split(' ', x), str_final))
        filtered_word = list(map(lambda x: ' '.join(list(np.array(x)[np.logical_not(np.in1d(x, stop_words))])), split_str_list))

        return filtered_word[0]
        #return jaso_split(filtered_word[0])


    def find_zero_texts(self, noun=False):
        self.texts = []
        self.new_clusterings = []
        for clustering in self.clusterings :
            if len(clustering['texts']) < 2 :
                text = clustering['texts'][0]
                convert_text = self.filtering(str_list=[text], noun=noun)
                self.texts.append({
                    'text' : convert_text,
                    'original' : text
                })
            else :
                self.new_clusterings.append(clustering)
        self.clusterings = self.new_clusterings

    def run(self):

        init_confidence = self.confidence
        self.init_run()
        print('finish init batch')

        self.confidence = init_confidence
        for i in range(self.batch_size) :
            print('ing verb batch >> ', i)
            if self.confidence >= 67 :
                self.run_batch(noun=True)
                self.run_batch(noun=False)
                self.confidence = self.confidence - 1.5
            elif self.confidence < 67 :
                break
        print('finish verb batch')


        ##그룹간 매칭
        new_small_c = []
        new_big_c = []
        for cluster in self.clusterings :
            if len(cluster['texts']) > 4 :
                new_big_c.append(cluster)
            else :
                new_small_c.append(cluster)

        print(new_small_c)
        max_sc_category = 0
        for sc in new_small_c :
            print('merge >> ', sc)
            max_ratio = 0
            max_bc_category = 0
            add_total_text = ""
            for bc in new_big_c :
                this_ratio = fuzz.token_set_ratio(sc['totalText'], bc['totalText'])
                if max_ratio < this_ratio :
                    max_bc_category = bc['category']
                    max_ratio = this_ratio

            if max_ratio > 70 :
                for item in new_big_c :
                    if item.get('category') == max_bc_category :
                        item['texts'].extend(sc['texts'])
                        item['totalText'] = item['totalText'] + sc['totalText']
                #[item['texts'].extend(sc['texts']) for item in new_big_c if item.get('category')==max_bc_category]

        self.clusterings = new_big_c

        return self.clusterings

    def init_run(self):
        for i, text in enumerate(self.before_texts):
            convert_text = self.filtering(str_list=[text], noun=True)
            if i == 0 and len(self.clusterings) == 0 :
                self.create_clustering(text, convert_text)
            else :
                self.ratio(text, convert_text)

    def run_batch(self, noun=True):

        # 독립적으로 묶여있는 클러스터링은 한번 더 돌려서 확인
        self.find_zero_texts(noun=noun)
        if len(self.texts) > 0 :
            for i, text_ob in enumerate(self.texts) :
                self.ratio(original=text_ob['original'], text=text_ob['text'])


    def ratio(self, original, text):
        max_category = 0
        max_ratio = 0
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

                cluster['totalText'] = cluster['totalText'] + ' ' + text
                tmp_totalTexts = list(set(re.split(' ', cluster['totalText'])))
                if len(tmp_totalTexts) < self.texts_len :
                    cluster['totalText'] = ' '.join(tmp_totalTexts)
                else :
                    count = Counter(tmp_totalTexts)
                    cluster['totalText'] = ""
                    for n, c in count.most_common(self.texts_len):
                        cluster['totalText'] = cluster['totalText'] + ' ' + n

    def convert_clustering(self, clusterings=[]):
        self.clusterings = []
        for c in clusterings :
            #print(c)
            tmp_total_str = ""
            totalText = ""
            tmp_total_original_list = []

            for t in c['texts'] :
                tmp_total_str = tmp_total_str + " " + self.filtering(str_list=[t], noun=False)
                tmp_total_original_list.append(t)

            tmp_total_list = re.split(' ', tmp_total_str)
            #print(tmp_total_list)
            if len(list(set(tmp_total_list))) > self.texts_len :
                count = Counter(tmp_total_list)
                for n, c in count.most_common(self.texts_len):
                    totalText = totalText + " " + n
            else :
                totalText = " ".join(list(set(tmp_total_list)))

            self.id = self.id + 1
            self.clusterings.append({
                'category' : self.id,
                'texts' : tmp_total_original_list,
                'totalText' : totalText
            })

if __name__ == "__main__":
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample2.txt"
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing.txt"  #1011
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing_cl.txt" #2650
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_bmt.txt"
    inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_samsung.txt"

    texts = []
    for uid, line in enumerate(open(inputfile)):
        texts.append(line.strip('\n'))

    random.shuffle(texts)


    import datetime

    s1 = datetime.datetime.now()

    #fw = FuzzyWuzzy(clusterings=[], texts=texts, confidence=0.8, batch_size=20)
    fw = FuzzyWuzzy(clusterings=[], texts=texts, confidence=0.8, batch_size=10)
    clusterings = fw.run()
    s2 = datetime.datetime.now()
    print(clusterings)
    print(s1, s2)
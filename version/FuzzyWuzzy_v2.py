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
# v1 -> v2
############
# 1. 보완 로직 추가
# 1-1.클러스터 add 될 때 마다 total text 값을 재 구성
#  -> 빈도수 측정 후 높은 빈도수 순서로 정해진 길이로 구성
# 1-2. 단건으로 묶여 있는 클러스터는 다시 한번 컨피던스 값을 살짝 낮춰서 재 구성
#  -> 컨피던스값을 낮춰도 되는 이유는 이미 앞에 돌때 단단하게? 클러스터링이 구성되어 있기 때문에~
# 2. clusterings 파라메타 추가
#  -> 기존 클러스터링이 있으면 추가로 텍스트들이 클러스터 될 수 있도록 클러스터링 파라메타 추가
#  -> 클러스터링 배열이 null이 아니면 비교할 수 있는 객체로 변환 시켜주는 로직 추가
############

class FuzzyWuzzy() :

    def __init__(self, clusterings=[], texts=[], confidence=0.6, batch_size=0) :
        self.texts = []
        self.batch_size = batch_size
        self.t = Twitter()
        self.confidence = confidence * 100
        self.clusterings = []
        self.id = 0
        self.texts_len = 13
        if len(clusterings) > 0 :
            self.convert_clustering(clusterings=clusterings)

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
        str_filt = np.array(str_pos)[np.where(np.in1d(list(map(itemgetter(1), str_pos)), ['Noun', 'Verb', 'Adjective', 'Alpha', 'Foreign', 'Number']))[0]]
        #str_filt = np.array(str_pos)[np.where(np.in1d(list(map(itemgetter(1), str_pos)), ['Noun', 'Alpha', 'Foreign', 'Number']))[0]]

        # if len(str_filt) > 1 and str_filt[-1][1] != 'Noun' :
        #     str_filt = str_filt[0:-1]

        str_final = [' '.join(list(map(itemgetter(0), str_filt)))]
        stop_words = ['가능하다','어떻다','어떤', '무슨', '알다', '말', '하다', '되다', '궁금하다', '가능하다', '시', '수', '인가요', '있다', '하나요', '해야하나요', '좋다', '해', '요', '한', '가요', '대해']
        split_str_list = list(map(lambda x: re.split(' ', x), str_final))

        filtered_word = list(map(lambda x: ' '.join(list(np.array(x)[np.logical_not(np.in1d(x, stop_words))])), split_str_list))

        return filtered_word[0]
        #return jaso_split(filtered_word[0])


    def run(self):

        for i, text_ob in enumerate(self.texts) :
            if i == 0 :
                self.create_clustering(text_ob['original'], text_ob['text'])
            else :
                self.ratio(text_ob)

        self.run_2(self.batch_size)

        return self.clusterings

    def run_2(self, size):
        # 독립적으로 묶여있는 클러스터링은 한번 더 돌려서 확인
        for i in range(size) :
            print(i)
            self.texts = []
            self.new_clusterings = []
            for clustering in self.clusterings :
                if len(clustering['texts']) < 2 :
                    text = clustering['texts'][0]
                    tmp_text = self.filtering([text])
                    self.texts.append({
                        'text' : tmp_text,
                        'original' : text
                    })
                else :
                    self.new_clusterings.append(clustering)

            if len(self.texts) > 0 :
                if self.confidence > 50 :
                    self.confidence = self.confidence - 1.5

                self.clusterings = self.new_clusterings
                for i, text_ob in enumerate(self.texts) :
                    self.ratio(text_ob)


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
                tmp_total_str = tmp_total_str + " " + self.filtering([t])
                tmp_total_original_list.append(t)

            tmp_total_list = re.split(' ', tmp_total_str)
            #print(tmp_total_list)
            if len(list(set(tmp_total_list))) > self.texts_len :
                count = Counter(tmp_total_list)
                for n, c in count.most_common(self.texts_len):
                    totalText = totalText + " " + n
            else :
                totalText = " ".join(tmp_total_list)

            self.id = self.id + 1
            self.clusterings.append({
                'category' : self.id,
                'texts' : tmp_total_original_list,
                'totalText' : totalText
            })

    def convert_return_dict(self):
        return


if __name__ == "__main__":
    # inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample2.txt"
    inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing.txt"  #1011
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing_cl.txt" #2650
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_bmt.txt"
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_samsung.txt"

    texts = []
    for uid, line in enumerate(open(inputfile)):
        texts.append(line.strip('\n'))

    # texts = texts[0:500]
    random.shuffle(texts)


    # num = 0
    # diff = 500
    # confidence = 0.7
    # cluterings = []
    # for i in  range( int(len(texts)/diff) ) :
    #     print(confidence)
    #     if i == 0 :
    #         ts = texts[i*diff:i*diff+diff]
    #         fw = FuzzyWuzzy(clusterings=[], texts=ts, confidence=confidence)
    #     else :
    #         ts = texts[i*diff+1:i*diff+diff]
    #         fw = FuzzyWuzzy(clusterings=cluterings, texts=ts, confidence=confidence)
    #     num = i
    #     confidence = confidence - 0.02
    #     cluterings = fw.run()
    # tsf = texts[num*diff+diff+1:]
    # FuzzyWuzzy(clusterings=cluterings, texts=tsf, confidence=confidence).run()
    # texts_1 = texts[0:2000]
    import datetime

    s1 = datetime.datetime.now()

    fw = FuzzyWuzzy(clusterings=[], texts=texts, confidence=0.8, batch_size=20)
    clusterings = fw.run()
    s2 = datetime.datetime.now()
    print(clusterings)
    print(s1, s2)
    #
    # texts_2 = texts[2000:]
    # fw = FuzzyWuzzy(clusterings=clusterings, texts=texts_2, confidence=0.7, batch_size=10)
    # clusterings_r = fw.run()
    # print(clusterings_r)
    # FuzzyWuzzy(clusterings=clusterings, texts=texts_2, confidence=0.7).run()





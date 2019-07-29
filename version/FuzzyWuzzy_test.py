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
        stop_words = ['가능하다','어떻다','어떤', '무슨', '알다', '말', '하다', '되다', '궁금하다', '가능하다', '시', '수', '인가요', '있다', '하나요', '해야하나요', '좋다', '해', '요', '한', '가요', '대해']
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

        # for i in range(2) :
        #     self.run_batch(noun=True)
        # print('finish noun batch')

        self.confidence = init_confidence
        for i in range(self.batch_size) :
            print('ing verb batch >> ', i)
            if self.confidence > 70 :
                self.run_batch(noun=True)
            self.run_batch(noun=False)
            if self.confidence > 50 :
                self.confidence = self.confidence - 1.5
        print('finish verb batch')

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

def filtering(str_list=None, noun=False):
    str_list = list(map(lambda x: re.sub('[\?\.]', '', x), str_list))
    str_list = list(map(lambda x: re.sub('어떻게 해야 하나요', '', x), str_list))
    str_list = list(map(lambda x: re.sub('어떻게 되는 것인가요', '', x), str_list))
    str_list = list(map(lambda x: re.sub('어떻게 되나요', '', x), str_list))
    str_list = list(map(lambda x: re.sub('왜 그런가요', '', x), str_list))
    str_list = list(map(lambda x: re.sub('라고 나와요', '', x), str_list))
    str_list = list(map(lambda x: re.sub('되나요', '', x), str_list))

    str_pos = Twitter().pos(str_list[0], stem=True)
    stop_pos = ['Noun', 'Alpha', 'Foreign', 'Number']
    if not(noun) :
        stop_pos.append('Verb')
        stop_pos.append('Adjective')

    str_filt = np.array(str_pos)[np.where(np.in1d(list(map(itemgetter(1), str_pos)), stop_pos))[0]]

    print(str_filt)

    str_final = [' '.join(list(map(itemgetter(0), str_filt)))]
    stop_words = ['가능하다','어떤', '무슨', '알다', '말', '하다', '되다', '궁금하다', '가능하다', '시', '수', '인가요', '있다', '하나요', '해야하나요', '좋다', '해', '요', '한', '가요', '대해']
    split_str_list = list(map(lambda x: re.split(' ', x), str_final))
    filtered_word = list(map(lambda x: ' '.join(list(np.array(x)[np.logical_not(np.in1d(x, stop_words))])), split_str_list))

    return filtered_word[0]

# t1 = filtering(str_list=["유심에 PUK번호가 어떤거에요?"], noun=False)
# t2 = filtering(str_list=["유심 어떻게 빼?"], noun=False)
# t3 = filtering(str_list=["유심에 PUK번호가 어떤거에요?"], noun=True)
# t4 = filtering(str_list=["유심 어떻게 빼?"], noun=True)
#
# print(t1)
# print(t2)
# print(t3)
# print(t4)
# a = fuzz.partial_ratio(t1,t2)
# b = fuzz.ratio(t1,t2)
# c = fuzz.token_set_ratio(t1,t2)
# d = fuzz.partial_token_set_ratio(t1,t2)
# e = fuzz.QRatio(t1,t2)
# f = fuzz.WRatio(t1,t2)
# g = fuzz.UWRatio(t1,t2)
# print(a,b,c,d,e,f,g)
s1 = filtering(str_list=["방카슈랑스 계약사항 중에서 변경하고 싶은 부분이 있습니다."], noun=False)
s2 = filtering(str_list=["스마트알림 메시지 데이터는 얼마동안 볼 수 있나요?"], noun=False)
s3 = "후 스마트 경우 수신 해외 서비스 외국 알림 신청 메시지 출국"
print(fuzz.token_set_ratio(s3, s2))
print(fuzz.QRatio(s3, s2))
print(fuzz.UWRatio(s3, s2))
print(fuzz.WRatio(s3, s2))
# testlist = [{
#     'category' : 1, 'value' : [1,2]
# },{
#     'category' : 2, 'value' : [1,2,3]
# }]
#
# print(testlist)
#
# print([value['value'].append(4) for value in testlist if value.get('category')==2])
#
# print(testlist)


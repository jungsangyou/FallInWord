import csv
from collections import Counter
import re



class FallInWord_Eval() :
    def __init__(self, clusterings=[], csvpath=None) :
        self.test_samples= []
        self.category_dict = {}
        f = open(csvpath, 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            try :
                self.category_dict[line[1]] = int(self.category_dict[line[1]]) + 1
            except :
                self.category_dict[line[1]] = 1

            self.test_samples.append({
                'category' : line[1],
                'text' : line[0]
            })
        f.close()
        self.clusterings = clusterings

    def run(self):
        for i, cluster in enumerate(self.clusterings) :
            texts = cluster['texts']
            for text in texts :
                for item in self.test_samples :
                    if item.get('text')==text :
                        try :
                            cluster['original_category'].append(item.get('category'))
                        except :
                            cluster['original_category'] = [item.get('category')]

        return self.get_point()

    def get_point(self):
        total_point = 0
        for i, cluster in enumerate(self.clusterings) :
            if cluster.get('original_category') != None :
                if len(cluster['texts']) > 1 :
                    count = Counter(cluster.get('original_category'))
                    length = len(cluster.get('original_category'))
                    for n, c in count.most_common(1):
                        #print(n, self.category_dict[n])
                        total_point = total_point + c/length
                        # total_point = total_point + c/self.category_dict[n]
        return total_point/len(self.clusterings)
#-*- coding: utf-8 -*-
from FallInWord import FallInWord
from FallInWord_Eval import FallInWord_Eval
import json, random
import datetime
from hanspell_custom import spell_checker


if __name__ == "__main__":
#     inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample2.txt" #2600, 430, #999, 197
#     inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing.txt"  #1011
#     inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_ing_cl.txt" #2650
#     inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/sample_bmt.txt" #
    #inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/file/sample_samsung.txt" #1216,101
#     inputfile = "/Users/j.s.you/Downloads/ING.txt" #1216,101
#     inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/file/shin_txt.txt" #1216,101

    inputfile = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/file/clms_test.txt" #1216,101


    #csvpath = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/file/testsample.csv" #1216,101
    # csvpath = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/file/shin_eval.csv" #1216,101
    csvpath = "/Users/j.s.you/Documents/workspace/dl/DL_B_Code/clustering/file/clms_result.csv" #1216,101


    texts = []
    for uid, line in enumerate(open(inputfile)):
        texts.append(line.strip('\n'))
    random.shuffle(texts)
    #texts = texts[0:100]
    #print(texts)
    s1 = datetime.datetime.now()
    fw = FallInWord(clusterings=[], texts=texts, confidence=0.8, batch_size=10, merge=False)
    clusterings = fw.run()
    s2 = datetime.datetime.now()

    print('[end] total process => ', s2-s1)
    print(clusterings)

    fw_e = FallInWord_Eval(clusterings=clusterings, csvpath=csvpath).run()

    result = {
        'acc' : fw_e,
        'clusterings' : clusterings
    }
    print(result)


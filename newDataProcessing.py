# 这个脚本将对CPED数据集进行处理，删除不用的信息（场景，说话人（将说话人归为A和B），情绪1，并将大五类分类）
#   TODO:
#       我们发现，CPED的数据集对话并不是一个每人说一句话的数据集
#       我们首先需要对数据进行处理
#
import csv

from tensorboardX import FileWriter
from translate import Translator

with open('../data/CPED/train_split.csv', encoding='utf-8') as file:
    reader = csv.reader(file)
    with open('../data/myCPED/new_myCPED.csv', 'w', newline='',encoding='utf-8') as f:
        myData = csv.writer(f)
        dialogue_ID = ''
        Utterance_ID = ''
        dict_emo = {}
        count = 0
        speaker = ''
        newRow = []
        emotion = ''
        context = ''
        cls = ''
        flag = 0
        for row in reader:
            if flag == 0:
                flag = 1
                continue
            if speaker == '':
                speaker = row[3]
            if dialogue_ID == '':
                dialogue_ID = row[1]
            if row[3] != speaker:
                count = count + 1
                newRow.append(speaker)
                newRow.append(count)
                if emotion in dict_emo.keys():
                    pass
                else:
                    dict_emo[emotion] = Translator(from_lang="English",to_lang="Chinese").translate(emotion)
                newRow.append(dict_emo[emotion])
                newRow.append(context)
                newRow.append(cls)
                myData.writerow(newRow)
                context = ''
                cls = ''
                newRow = []
            speaker = row[3]
            emotion = row[15]
            context = context + row[17] + '。'
            cls = ''
            for i in range(6, 11):
                if row[i] == 'high':
                    cls = cls + '1'
                elif row[i] == 'low':
                    cls = cls + '0'
            if len(cls) < 5:
                cls = 'UNK'
            if dialogue_ID != row[1]:
                dialogue_ID = row[1]
                count = 0
        f.close()
    file.close()

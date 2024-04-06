# 这个脚本将对CPED数据集进行处理，删除不用的信息（场景，说话人（将说话人归为A和B），情绪1，并将大五类分类）
#   TODO:
#       我们发现，CPED的数据集对话并不是一个每人说一句话的数据集
#       我们首先需要对数据进行处理
#
import csv

from tensorboardX import FileWriter

with open('../data/CPED/train_split.csv', encoding='utf-8') as file:
    reader = csv.reader(file)
    with open('../data/myCPED/myCped.csv', 'w', newline='',encoding='utf-8') as f:
        myData = csv.writer(f)
        Dialogue_ID = ''
        Utterance_ID = ''
        count = 0
        SpeakerA = ''
        SpeakerB = ''
        for row in reader:
            newRow = []
            if Dialogue_ID != row[1]:
                Dialogue_ID = row[1]
                count = 0
                SpeakerA = ''
                SpeakerB = ''
            # --------给说话者A和B赋值------------
            if SpeakerA == '':
                SpeakerA = row[3]
            if SpeakerB == '' and SpeakerA != row[3]:
                SpeakerB = row[3]
            # --------------判断当前说话者是A还是B----------------
            if SpeakerA == row[3]:
                newRow.append(SpeakerA)
            elif SpeakerB == row[3]:
                newRow.append(SpeakerB)
            else:
                newRow.append(row[3])
            # ------------------对话轮次赋值----------------
            count = count + 1
            newRow.append(count)
            # ------------------对话情绪及说话内容赋值----------------
            newRow.append(row[15])
            newRow.append(row[17])
            # ------------------将分类写入文件----------------
            cls = ''
            for i in range(6, 11):
                if row[i] == 'high':
                    cls = cls + '1'
                elif row[i] == 'low':
                    cls = cls + '0'
            if len(cls) < 5:
                cls = 'UNK'
            newRow.append(cls)
            # ------------将newRow写入csv文件------------------------
            myData.writerow(newRow)

import csv
import itertools

# 生成二进制数列表
binary_numbers = ['0', '1']
#all_list将所有人分成32类
# 使用permutations函数生成所有排列
permutations_list = list(itertools.permutations(binary_numbers, 5))
dict = {}
all = {}
all_list = []
with open('../data/CPED/train_split.csv', encoding='utf-8') as file:
    with open('../data/myCPED/colorMotif.csv', 'w', newline='',encoding='utf-8') as f:
        reader = csv.reader(file)
        writer = csv.writer(f)
        for row in reader:
            cls = ''
            all = {}
            newRow = []
            for i in range(6, 11):
                if row[i] == 'high':
                    cls = cls + '1'
                elif row[i] == 'low':
                    cls = cls + '0'
            if len(cls) < 5:
                cls = 'UNK'
            newRow.append(cls)
            newRow.append(row[14])
            newRow.append(row[15])
            all_list.append(newRow)
        writer.writerows(all_list)
        f.close()
        file.close()



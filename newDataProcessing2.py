import csv

#
# with open('../data/CPED/train_split.csv', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     tv_id = ''
#     count = 1
#     tv_dict = {}
#     tv_idlist = []
#     flag = 0
#     context = ''
#     for row in reader:
#         if flag == 0:
#             flag = 1
#             continue
#         if tv_id == '':
#             tv_id = row[1]
#         if row[1] == tv_id:
#             count = count + 1
#         else:
#             tv_id = row[1]
#             if count in tv_dict.keys():
#                 tv_idlist = tv_dict[count]
#                 tv_idlist.append(row[1])
#                 tv_dict[count] = tv_idlist
#             else:
#                 tv_idlist = []
#                 tv_idlist.append(row[1])
#                 tv_dict[count] = tv_idlist
#             count = 1
#
#     for i in tv_dict:
#         print(i,len(tv_dict[i]))
#
#
#     file.close()
#

with open('../data/CPED/train_split.csv', encoding='utf-8') as file:
    reader = csv.reader(file)
    tv_id = ''
    content = ''
    flag = 0
    count = 1
    count_small = 0
    new_list = []
    for row in reader:
        if flag == 0:
            flag = 1
            continue
        if tv_id == '':
            tv_id = row[1]

        if row[1] == tv_id:
            content = content + row[17]
        else:
            if 200 > len(content) > 40:
                print(len(content))
                count_small = count_small + 1
            else:
                new_list.append(tv_id)
                print("tv_id:" +tv_id,"len:"+ str(len(content)))
            tv_id = row[1]
            content = ''
            count = count+1

    print('总共:' + str(count),'小于220：' + str(count_small))
    print(new_list)
    file.close()
i = 0
with open('../data/CPED/new_train_split.csv', "w",encoding='utf-8',newline='') as f:
    with open('../data/CPED/train_split.csv', encoding='utf-8') as p:
        reader = csv.reader(p)
        my_writer = csv.writer(f)
        for row in reader:
            i = i + 1
            if row[1] not in new_list:
                my_writer.writerow(row)
print('新数据的对话量数：' + str(i))

p.close()
f.close()


with open('../data/CPED/new_train_split.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        count = count + 1


    print('新数据的对话量数：'+ str(count))
    f.close()

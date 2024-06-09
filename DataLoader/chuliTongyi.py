import csv




with open('./myCPEDTongyi.csv', encoding='utf-8') as file:
    with open('./myCPEDTongyichuli.csv', 'w',encoding='utf-8') as p:
        reader = csv.reader(file)
        t = csv.writer(p)
        for row in reader:
            if row == '':
                continue
            t.writerow(row)
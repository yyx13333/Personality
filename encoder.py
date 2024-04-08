import csv
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertTokenizer, \
    BertModel
from translate import Translator  # noqa

# --------------------------加载模型---------------------------------------
# config = AutoConfig.from_pretrained("../../bert/config.json")
# model = AutoModelForMaskedLM.from_pretrained("../../bert/pytorch_model.bin", config=config)
# tokenizer = AutoTokenizer.from_pretrained("../../bert/",
#                                           config="../../bert/tokenizer_config.json",
#                                           vocab_file="../../bert/vocab.txt")
config = AutoConfig.from_pretrained("../bert/config.json")
tokenizer = BertTokenizer.from_pretrained("../bert/",
                                          config="../bert/tokenizer_config.json",
                                          vocab_file="../bert/vocab.txt")
model = BertModel.from_pretrained("../bert/pytorch_model.bin", config=config)

# -------------------------打开myCPED.csv文件------------------------------
with open("../data/myCPED/tarin_data_bert.json.feature_emo0", "w", encoding="utf-8") as file:
    with open("../data/myCPED/new_myCPED.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        count = 0
        diglog = []
        all = []
        for row in reader:
            if int(row[1]) > count:
                count = int(row[1])
            else:
                count = 1
                print(diglog)
                all.append(diglog)
                diglog = []
            input_text = '以一种'+row[2]+"的情绪说："+row[3]
            # 使用tokenizer对输入文本进行编码
            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True)
            output = model(**input_ids).pooler_output.tolist()
            dict = {}
            dict['text'] = row[3]
            dict['speaker'] = row[0]
            dict['cls'] = output[0]
            if row[4] == 'UNK':
                dict['label'] = row[4]
            else:
                dict['label'] = row[4][0]
            dict['ent'] = row[2]
            diglog.append(dict)

        file.write(json.dumps(all, ensure_ascii=False))
    f.close()
    file.close()

# with open("../data/myCPED/tarin_data_bert.json.feature", encoding="utf-8") as file:
#     raw_data = json.load(file)
#     for r in raw_data:
#         print(r)

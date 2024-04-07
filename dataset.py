import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CPEDDataset(Dataset):

    def __init__(self, dataset_name="CPED", split="train", speaker_vocab=None, label_vocab=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.data = self.read(dataset_name, split, None)
        print(len(self.data))
        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open("../data/myCPED/tarin_data_bert.json.feature_emo0", encoding="utf-8") as f:
            raw_data = json.load(f)

        diglogs = []
        for d in raw_data:
            uterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                uterances.append(u["text"])
                labels.append(self.label_vocab['stoi'][u["label"]] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])

            diglogs.append({
                "utterances": uterances,
                "labels": labels,
                "speakers": speakers,
                "features": features
            })
        random.shuffle(diglogs)
        return diglogs

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]["features"]), \
            torch.LongTensor(self.data[index]["labels"]), \
            self.data[index]['speakers'], \
            len(self.data[index]['labels']), \
            self.data[index]['utterances']

    def get_adj(self, speakers, max_dialog_len):
        '''
        get adj matrix
        :param speaker:
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                get_local_pred = False
                get_global_pred = False
                for j in range(i - 1, -1, -1):
                    if speaker[j] == s and not get_local_pred:
                        get_local_pred = True
                        a[i, j] = 1
                    elif speaker[j] != s and not get_global_pred:
                        get_global_pred = True
                        a[i, j] = 1
                    if get_global_pred and get_global_pred:
                        break
            adj.append(a)
        return torch.stack(adj)

    def get_adj_v1(self, speakers, max_dialog_len):
        adj = []
        for speaker in speakers:
            a = torch.zeros(max_dialog_len, max_dialog_len)
            for i, s in enumerate(speaker):
                cnt = 0
                for j in range(i - 1, -1, -1):
                    a[i, j] = 1
                    if speaker[j] == s:
                        cnt += 1
                        if cnt == 2:
                            break
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        s_mask = []
        s_mask_onehot = []
        for speaker in speakers:
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype=torch.long)
            s_onehot = torch.zeros(max_dialog_len, max_dialog_len, 2)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1
                        s_onehot[i, j, 1] = 1
                    else:
                        s_onehot[i, j, 0] = 1
            s_mask.append(s)
            s_mask_onehot.append(s_onehot)
        return torch.stack(s_mask), torch.stack(s_mask_onehot)

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_diglog_len = max([d[3] for d in data])
        features = pad_sequence([d[0] for d in data], batch_first=True)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)
        adj = self.get_adj_v1([d[2] for d in data], max_diglog_len)
        s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_diglog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)
        utterances = [d[4] for d in data]
        return features, labels, adj, s_mask, s_mask_onehot, lengths, speakers, utterances

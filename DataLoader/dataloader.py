import pickle

from torch.utils.data import SubsetRandomSampler, DataLoader

from dataset import *

from myProject1.dataset import CPEDDataset


def get_train_valid_sample(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)


def load_vocab():
    speaker_vocab = pickle.load(open('../data/myCPED/speaker_vocab.pkl', 'rb'))
    label_vocab = pickle.load(open('../data/myCPED/label_vocab.pkl', 'rb'))
    person_vec = None
    return speaker_vocab, label_vocab, person_vec


def get_CPED_loaders(dataset_name, batch_size, num_workers):
    print("building vocab...")
    speaker_vocab, label_vocab, person_vec = load_vocab()
    print("building datasets...")
    trainset = CPEDDataset("CPED", "train", speaker_vocab, label_vocab)
    train_sampler = get_train_valid_sample(trainset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=False
                              )
    return train_loader, speaker_vocab, label_vocab, person_vec

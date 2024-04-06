import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import writer


def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, args, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    cnt = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        features, label, adj, s_mask, s_mask_onehot, lengths, speakers, utterances = data
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask = s_mask.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            lengths = lengths.cuda()

        # print(speakers)
        log_prob = model(features, adj, s_mask, s_mask_onehot, lengths)  # (B, N, C)
        # print(label)
        loss = loss_function(log_prob.permute(0, 2, 1), label)

        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim=2).cpu().numpy().tolist()
        preds += pred
        labels += label
        losses.append(loss.item())

        if train:
            loss_val = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        new_preds = []
        new_labels = []
        for i, label in enumerate(labels):
            for j, l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    # print(preds.tolist())
    # print(labels.tolist())
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
    avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, avg_micro_fscore, avg_macro_fscore

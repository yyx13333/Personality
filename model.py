import torch
from transformers import BertModel, BertConfig

from model_utils import *


class DAGERC_fushion(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                # gats += [GAT_dialoggcn(args.hidden_dim)]
                gats += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)

        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        self.nodal_att_type = args.nodal_att_type

        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim

        # output mlp layers
        layers = [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]
        self.out_mlp = nn.Sequential(*layers)

        self.pc = nn.Linear(in_dim, args.hidden_dim)
        self.attentive_node_features = attentive_node_features(in_dim)
        self.lstmA = nn.LSTM(args.hidden_dim,
                            args.hidden_dim,
                            bidirectional=False,
                            batch_first=True,
                            dropout=args.dropout
                            )
        self.lstmB = nn.LSTM(args.hidden_dim,
                            args.hidden_dim,
                            bidirectional=False,
                            batch_first=True,
                            dropout=args.dropout
                            )

    def forward(self, features, adj, s_mask, s_mask_onehot, lengths,t):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        # H0 = self.dropout(H0)
        H = [H0]
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:, 0, :]).unsqueeze(1)
            M = torch.zeros_like(C).squeeze(1)
            # P = M.unsqueeze(1)
            P = self.grus_p[l](M, H[l][:, 0, :]).unsqueeze(1)
            # H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))
            # H1 = F.relu(C+P)
            H1 = C + P
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i], s_mask[:, i, :i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i])
                    else:
                        _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i], s_mask[:, i, :i])

                C = self.grus_c[l](H[l][:, i, :], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:, i, :]).unsqueeze(1)
                # P = M.unsqueeze(1)
                # H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))
                # H_temp = F.relu(C+P)
                H_temp = C + P
                H1 = torch.cat((H1, H_temp), dim=1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
        H.append(features)

        H = torch.cat(H, dim=2)

        H = self.attentive_node_features(H, lengths, self.nodal_att_type)
        H = self.pc(H)
        A = torch.zeros(H.shape[0], H.shape[1], H.shape[2])
        B = torch.zeros(H.shape[0], H.shape[1], H.shape[2])
        for i in range(H.shape[0]):     #维度直接相加
            i_c = -1
            for i_t in t[i]:
                i_c = i_c + 1
                if i_t == '1':
                    B[i][i_c][:] = H[i][i_c][:]
                else:
                    A[i][i_c][:] = H[i][i_c][:]

        A,_ = self.lstmA(A)
        B,_ = self.lstmB(B)

        A = F.adaptive_avg_pool1d(A.permute(0,2,1),output_size=1).permute(0,2,1)
        B = F.adaptive_avg_pool1d(B.permute(0, 2, 1), output_size=1).permute(0, 2, 1)
        H = torch.cat((A,B),dim=1)
#-----------------------------------------------------------------------
        # T = torch.zeros(H.shape[0], 2, H.shape[2])
        #
        # for i in range(H.shape[0]):     #维度直接相加
        #     i_c = -1
        #     for i_t in t[i]:
        #         i_c = i_c + 1
        #         if i_t == '1':
        #             T[i][0][:] += H[i][i_c][:]
        #         else:
        #             T[i][1][:] += H[i][i_c][:]
#----------------------------------------------------------------------
        logits = self.out_mlp(H)

        return logits


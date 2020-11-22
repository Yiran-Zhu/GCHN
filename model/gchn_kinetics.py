import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from .gchb_kinetics import GCHB_4Stages_withTCN as GCHB
from .conv_module import unit_gcn, unit_tcn
from graph.kinetics_multiscale import Graph_J, Graph_P, Graph_B
from .conv_module import bn_init

'''
The Implementation of Graph Convolutional Hourglass Network for Kinetics-Skeleton.
'''

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

class GCHB_module(nn.Module):
    def __init__(self, dim, A, B, C, stride=1, residual=True):
        super(GCHB_module, self).__init__()
        self.gcn1 = GCHB(dim, A, B, C)
        self.tcn1 = unit_tcn(dim, dim, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif stride == 1:
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(dim, dim, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3, mid_supervise=False):
        super(Model, self).__init__()
        self.graph_j = Graph_J()
        self.graph_p = Graph_P()
        self.graph_b = Graph_B()
        A = self.graph_j.A_j
        B = self.graph_p.A_p
        C = self.graph_b.A_b

        self.mid_supervise = mid_supervise

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.ST_GConv1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.GCHB_m1 = GCHB_module(64, A, B, C)
        self.ST_GConv2 = TCN_GCN_unit(64, 128, A, stride=2)
        self.GCHB_m2 = GCHB_module(128, A, B, C)
        self.ST_GConv3 = TCN_GCN_unit(128, 256, A, stride=2)
        self.GCHB_m3 = GCHB_module(256, A, B, C)
        self.ST_GConv4 = TCN_GCN_unit(256, 256, A)
        if self.mid_supervise:
            self.fc1 = nn.Linear(64, num_class)
            self.fc2 = nn.Linear(128, num_class)
            nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / num_class))
            nn.init.normal_(self.fc2.weight, 0, math.sqrt(2. / num_class))
        self.fc3 = nn.Linear(256, num_class)
    
        nn.init.normal_(self.fc3.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        if not self.mid_supervise:
            x = self.ST_GConv1(x)
            x = self.GCHB_m1(x)
            x = self.ST_GConv2(x)
            x = self.GCHB_m2(x)
            x = self.ST_GConv3(x)
            x = self.GCHB_m3(x)
            x = self.ST_GConv4(x)
            # N*M,C,T,V
            c_new = x.size(1)
            x = x.view(N, M, c_new, -1)
            x = x.mean(3).mean(1)

            return self.fc3(x)
        else:
            # Stem.
            x = self.ST_GConv1(x)
            # Hourglass.
            x = self.GCHB_m1(x)
            out1 = self.ST_GConv2(x)
            x = self.GCHB_m2(out1)
            out2 = self.ST_GConv3(x)
            x = self.GCHB_m3(out2)
            out3 = self.ST_GConv4(x)
            # N*M,C,T,V
            out = []
            for x in [out1, out2, out3]:
                c_new = x.size(1)
                x = x.view(N, M, c_new, -1)
                x = x.mean(3).mean(1)
                out.append(x)
            return out

        

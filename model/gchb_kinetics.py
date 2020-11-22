import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import unit_gcn as GCN
from .conv_module import unit_tcn as TCN

'''
The Implementation of Graph Convolutional Hourglass Block (GCHB Module) for Kinetics-Skeleton.
'''

class Down_Joint2Part(nn.Module):
    def __init__(self):
        super().__init__()
        self.torso = [0,1]
        self.left_leg_up = [8]
        self.left_leg_down = [9,10]
        self.right_leg_up = [11]
        self.right_leg_down = [12,13]
        self.head_left = [14,16]
        self.head_right = [15,17]
        self.left_arm_up = [3,4]
        self.left_arm_down = [2]
        self.right_arm_up = [6,7]
        self.right_arm_down = [5]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))                                              
        x_leftlegup = x[:, :, :, self.left_leg_up]                              
        x_leftlegdown = F.avg_pool2d(x[:, :, :, self.left_leg_down], kernel_size=(1, 2))                  
        x_rightlegup = x[:, :, :, self.right_leg_up]                       
        x_rightlegdown = F.avg_pool2d(x[:, :, :, self.right_leg_down], kernel_size=(1, 2))                   
        x_head_left = F.avg_pool2d(x[:, :, :, self.head_left], kernel_size=(1, 2))
        x_head_right = F.avg_pool2d(x[:, :, :, self.head_right], kernel_size=(1, 2))                                              
        x_leftarmup = F.avg_pool2d(x[:, :, :, self.left_arm_up], kernel_size=(1, 2))                            
        x_leftarmdown = x[:, :, :, self.left_arm_down]                
        x_rightarmup = F.avg_pool2d(x[:, :, :, self.right_arm_up], kernel_size=(1, 2))                       
        x_rightarmdown =  x[:, :, :, self.right_arm_down]             
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head_left, x_head_right,  x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)            
        return x_part


class Down_Part2Body(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [4,5,6]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [7,8]
        self.right_arm = [9,10]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 3))                              
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 2))                
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 2))        
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 2))            
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 2))        
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)               
        return x_body



class Up_Part2Joint(nn.Module):

    def __init__(self):
        super().__init__()
        # for all: index - 1
        self.torso = [0,1]
        self.left_leg_up = [8]
        self.left_leg_down = [9,10]
        self.right_leg_up = [11]
        self.right_leg_down = [12,13]
        self.head_left = [14,16]
        self.head_right = [15,17]
        self.left_arm_up = [3,4]
        self.left_arm_down = [2]
        self.right_arm_up = [6,7]
        self.right_arm_down = [5]

    def forward(self, part):
        N, d, T, w = part.size()  # [64, 256, 7, 10]
        x = part.new_zeros((N, d, T, 18))

        x[:,:,:,self.left_leg_up] = part[:,:,:,0].unsqueeze(-1)
        x[:,:,:,self.left_leg_down] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_up] = part[:,:,:,2].unsqueeze(-1)
        x[:,:,:,self.right_leg_down] = torch.cat((part[:,:,:,3].unsqueeze(-1), part[:,:,:,3].unsqueeze(-1)),-1)
        x[:,:,:,self.torso] = torch.cat((part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1)),-1)
        x[:,:,:,self.head_left] = torch.cat((part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1)),-1)
        x[:,:,:,self.head_right] = torch.cat((part[:,:,:,6].unsqueeze(-1), part[:,:,:,6].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_up] = torch.cat((part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_down] = part[:,:,:,8].unsqueeze(-1)
        x[:,:,:,self.right_arm_up] = torch.cat((part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_down] = part[:,:,:,10].unsqueeze(-1)

        return x


class Up_Body2Part(nn.Module):

    def __init__(self):
        super().__init__()

        self.torso = [4,5,6]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [7,8]
        self.right_arm = [9,10]

    def forward(self, body):
        N, d, T, w = body.size()  
        x = body.new_zeros((N, d, T, 11))

        x[:,:,:,self.left_leg] = torch.cat((body[:,:,:,0:1], body[:,:,:,0:1]),-1)
        x[:,:,:,self.right_leg] = torch.cat((body[:,:,:,1:2], body[:,:,:,1:2]),-1)
        x[:,:,:,self.torso] = torch.cat((body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3]),-1)
        x[:,:,:,self.left_arm] = torch.cat((body[:,:,:,3:4], body[:,:,:,3:4]),-1)
        x[:,:,:,self.right_arm] = torch.cat((body[:,:,:,4:5], body[:,:,:,4:5]),-1)

        return x


class Graph_UNet_Unit(nn.Module):
    def __init__(self, dim, A, B, C):
        super(Graph_UNet_Unit, self).__init__()
        # GraphConv.
        self.GCN_J = GCN(dim, dim, A)
        self.Up_GCN_J = GCN(2*dim, dim, A)
        self.GCN_P = GCN(dim, dim, B)
        self.Up_GCN_P = GCN(2*dim, dim, B)
        self.GCN_B = GCN(dim, dim, C)
        # Down and Up.
        self.Down_J2P = Down_Joint2Part()
        self.Down_P2B = Down_Part2Body()
        self.Up_B2P = Up_Body2Part()
        self.Up_P2J = Up_Part2Joint()

    def forward(self, J_in):
        J_pre = self.GCN_J(J_in)
        down1 = self.Down_J2P(J_pre)
        P_pre = self.GCN_P(down1)
        down2 = self.Down_P2B(P_pre)
        B_pre = self.GCN_B(down2)
        up1 = self.Up_B2P(B_pre)
        P_next = self.Up_GCN_P(torch.cat((P_pre, up1), dim=1))
        up2 = self.Up_P2J(P_next)
        J_next = self.Up_GCN_J(torch.cat((J_pre, up2), dim=1))

        return J_next

class Down_Body2Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.node = [0,1,2,3,4]
    def forward(self, x):
        x_node = F.avg_pool2d(x[:,:,:,self.node], kernel_size=(1, 5))
        return x_node

class up_Node2Body(nn.Module):
    def __init__(self):
        super().__init__()

        self.node = [0,1,2,3,4]

    def forward(self, node):
        N, d, T, w = node.size() 
        x = node.new_zeros((N, d, T, 5))

        x[:,:,:,self.node] = torch.cat((node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1]),-1)

        return x

class GCHB_4Stages_withTCN(nn.Module):
    def __init__(self, dim, A, B, C):
        super(GCHB_4Stages_withTCN, self).__init__()
        # GraphConv.
        self.GCN_J = GCN(dim, dim, A)
        self.Up_GCN_J = GCN(2*dim, dim, A)
        self.GCN_P = GCN(dim, dim, B)
        self.Up_GCN_P = GCN(2*dim, dim, B)
        self.GCN_B = GCN(dim, dim, C)
        self.Up_GCN_B = GCN(2*dim, dim, C)
        # TemporalConv.
        self.TCN = TCN(dim, dim, kernel_size=9, stride=1)
        # Down and Up.
        self.Down_J2P = Down_Joint2Part()
        self.Down_P2B = Down_Part2Body()
        self.Down_B2N = Down_Body2Node()
        self.Up_B2P = Up_Body2Part()
        self.Up_P2J = Up_Part2Joint()
        self.Up_N2B = up_Node2Body()

    def forward(self, J_in):
        J_pre = self.GCN_J(J_in)
        down1 = self.Down_J2P(J_pre)
        P_pre = self.GCN_P(down1)
        down2 = self.Down_P2B(P_pre)
        B_pre = self.GCN_B(down2)
        down3 = self.Down_B2N(B_pre)
        Node_pre = self.TCN(down3)
        up0 = self.Up_N2B(Node_pre)
        B_next = self.Up_GCN_B(torch.cat((B_pre, up0), dim=1))
        up1 = self.Up_B2P(B_next)
        P_next = self.Up_GCN_P(torch.cat((P_pre, up1), dim=1))
        up2 = self.Up_P2J(P_next)
        J_next = self.Up_GCN_J(torch.cat((J_pre, up2), dim=1))

        return J_next
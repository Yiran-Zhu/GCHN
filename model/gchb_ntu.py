import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import unit_gcn as GCN
from .conv_module import unit_tcn as TCN

'''
The Implementation of Graph Convolutional Hourglass Block (GCHB Module) for NTU-RGB+D.
'''

class Down_Joint2Part(nn.Module):
    def __init__(self):
        super().__init__()
        self.torso = [0,1,20]
        self.left_leg_up = [16,17]
        self.left_leg_down = [18,19]
        self.right_leg_up = [12,13]
        self.right_leg_down = [14,15]
        self.head = [2,3]
        self.left_arm_up = [8,9]
        self.left_arm_down = [10,11,23,24]
        self.right_arm_up = [4,5]
        self.right_arm_down = [6,7,21,22]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 3))                                             
        x_leftlegup = F.avg_pool2d(x[:, :, :, self.left_leg_up], kernel_size=(1, 2))                               
        x_leftlegdown = F.avg_pool2d(x[:, :, :, self.left_leg_down], kernel_size=(1, 2))                    
        x_rightlegup = F.avg_pool2d(x[:, :, :, self.right_leg_up], kernel_size=(1, 2))                       
        x_rightlegdown = F.avg_pool2d(x[:, :, :, self.right_leg_down], kernel_size=(1, 2))                  
        x_head = F.avg_pool2d(x[:, :, :, self.head], kernel_size=(1, 2))                                    
        x_leftarmup = F.avg_pool2d(x[:, :, :, self.left_arm_up], kernel_size=(1, 2))                    
        x_leftarmdown = F.avg_pool2d(x[:, :, :, self.left_arm_down], kernel_size=(1, 4))              
        x_rightarmup = F.avg_pool2d(x[:, :, :, self.right_arm_up], kernel_size=(1, 2))                
        x_rightarmdown = F.avg_pool2d(x[:, :, :, self.right_arm_down], kernel_size=(1, 4))            
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head,  x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)            
        return x_part


class Down_Part2Body(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [4,5]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [6,7]
        self.right_arm = [8,9]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))                                           
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 2))                          
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 2))                       
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 2))                           
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 2))                       
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)              
        return x_body

class Down_Body2Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.node = [0,1,2,3,4]
    def forward(self, x):
        x_node = F.avg_pool2d(x[:,:,:,self.node], kernel_size=(1, 5))
        return x_node


class Up_Part2Joint(nn.Module):

    def __init__(self):
        super().__init__()
        # for all: index - 1
        self.torso = [0,1,20]
        self.left_leg_up = [16,17]
        self.left_leg_down = [18,19]
        self.right_leg_up = [12,13]
        self.right_leg_down = [14,15]
        self.head = [2,3]
        self.left_arm_up = [8,9]
        self.left_arm_down = [10,11,23,24]
        self.right_arm_up = [4,5]
        self.right_arm_down = [6,7,21,22]

    def forward(self, part):
        N, d, T, w = part.size()  
        x = part.new_zeros((N, d, T, 25))

        x[:,:,:,self.left_leg_up] = torch.cat((part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1)),-1)
        x[:,:,:,self.left_leg_down] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_up] = torch.cat((part[:,:,:,2].unsqueeze(-1), part[:,:,:,2].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_down] = torch.cat((part[:,:,:,3].unsqueeze(-1), part[:,:,:,3].unsqueeze(-1)),-1)
        x[:,:,:,self.torso] = torch.cat((part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1)),-1)
        x[:,:,:,self.head] = torch.cat((part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_up] = torch.cat((part[:,:,:,6].unsqueeze(-1),part[:,:,:,6].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_down] = torch.cat((part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_up] = torch.cat((part[:,:,:,8].unsqueeze(-1),part[:,:,:,8].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_down] = torch.cat((part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1)),-1)

        return x


class Up_Body2Part(nn.Module):

    def __init__(self):
        super().__init__()

        self.torso = [4,5]
        self.left_leg = [0,1]
        self.right_leg = [2,3]
        self.left_arm = [6,7]
        self.right_arm = [8,9]

    def forward(self, body):
        N, d, T, w = body.size()
        x = body.new_zeros((N, d, T, 10))

        x[:,:,:,self.left_leg] = torch.cat((body[:,:,:,0:1], body[:,:,:,0:1]),-1)
        x[:,:,:,self.right_leg] = torch.cat((body[:,:,:,1:2], body[:,:,:,1:2]),-1)
        x[:,:,:,self.torso] = torch.cat((body[:,:,:,2:3], body[:,:,:,2:3]),-1)
        x[:,:,:,self.left_arm] = torch.cat((body[:,:,:,3:4], body[:,:,:,3:4]),-1)
        x[:,:,:,self.right_arm] = torch.cat((body[:,:,:,4:5], body[:,:,:,4:5]),-1)

        return x


class up_Node2Body(nn.Module):
    def __init__(self):
        super().__init__()

        self.node = [0,1,2,3,4]

    def forward(self, node):
        N, d, T, w = node.size()
        x = node.new_zeros((N, d, T, 5))

        x[:,:,:,self.node] = torch.cat((node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1], node[:,:,:,0:1]),-1)

        return x


class GCHB_4Stages(nn.Module):
    def __init__(self, dim, A, B, C):
        super(GCHB_4Stages, self).__init__()
        # GraphConv.
        self.GCN_J = GCN(dim, dim, A)
        self.Up_GCN_J = GCN(2*dim, dim, A)
        self.GCN_P = GCN(dim, dim, B)
        self.Up_GCN_P = GCN(2*dim, dim, B)
        self.GCN_B = GCN(dim, dim, C)
        self.Up_GCN_B = GCN(2*dim, dim, C)
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
        up0 = self.Up_N2B(down3)
        B_next = self.Up_GCN_B(torch.cat((B_pre, up0), dim=1))
        up1 = self.Up_B2P(B_next)
        P_next = self.Up_GCN_P(torch.cat((P_pre, up1), dim=1))
        up2 = self.Up_P2J(P_next)
        J_next = self.Up_GCN_J(torch.cat((J_pre, up2), dim=1))

        return J_next


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


class GCHB_3Stages(nn.Module):
    def __init__(self, dim, A, B, C):
        super(GCHB_3Stages, self).__init__()
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


class GCHB_2Scales(nn.Module):
    def __init__(self, dim, A, B, C):
        super(GCHB_2Scales, self).__init__()
        # GraphConv.
        self.GCN_J = GCN(dim, dim, A)
        self.Up_GCN_J = GCN(2*dim, dim, A)
        self.GCN_P = GCN(dim, dim, B)
        # Down and Up.
        self.Down_J2P = Down_Joint2Part()
        self.Up_P2J = Up_Part2Joint()

    def forward(self, J_in):
        J_pre = self.GCN_J(J_in)
        down = self.Down_J2P(J_pre)
        P_pre = self.GCN_P(down)
        up = self.Up_P2J(P_pre)
        J_next = self.Up_GCN_J(torch.cat((J_pre, up), dim=1))

        return J_next
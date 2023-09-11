from cmath import nan
import torch
import numpy as np
# from Track1.utils import *
from utils import *
import math

# only loss function
# def contrastive_loss(x1,x2,kinship,race,bias_map,bias_pair,beta=0.08):
#     x1x2=torch.cat([x1,x2],dim=0)
#     x2x1=torch.cat([x2,x1],dim=0)
    
#     cosine_mat=torch.cosine_similarity(torch.unsqueeze(x1x2,dim=1),
#                                        torch.unsqueeze(x1x2,dim=0),dim=2)/(beta)
#     mask=1.0-torch.eye(2*x1.size(0))
#     diagonal_cosine=torch.cosine_similarity(x1x2,x2x1,dim=1)
    
#     debais_margin=torch.sum(bias_map,axis=1)/len(bias_map)
#     numerators = torch.exp((diagonal_cosine-debais_margin)/(beta))
#     denominators=torch.sum(torch.exp(cosine_mat)*mask,dim=1)-torch.exp(diagonal_cosine/beta)+numerators # - x1x2/beta+ x1x2/beta+epsilon

#     return -torch.mean(torch.log(numerators)-torch.log(denominators),dim=0), [AA_margin, A_margin, C_margin, I_margin]

# loss function with bias of every race
def contrastive_loss(x1,x2,kinship,race,bias_map,bias_pair,beta=0.08):

    batch_size = x1.size(0)
    AA_num = 0
    A_num = 0
    C_num = 0
    I_num = 0
    AA_idx = 0
    A_idx = 0
    C_idx = 0
    I_idx = 0
    x1x2=torch.cat([x1,x2],dim=0)
    x2x1=torch.cat([x2,x1],dim=0)
    
    cosine_mat=torch.cosine_similarity(torch.unsqueeze(x1x2,dim=1),
                                       torch.unsqueeze(x1x2,dim=0),dim=2)/(beta)
    mask=1.0-torch.eye(2*x1.size(0))
    diagonal_cosine=torch.cosine_similarity(x1x2,x2x1,dim=1)
    
    debais_margin=torch.sum(bias_map,axis=1)/len(bias_map)
    for i in range(batch_size):
        if race[i] == 0:
            AA_num += debais_margin[i] + debais_margin[i+batch_size]
            AA_idx += 2
        elif race[i] == 1:
            A_num += debais_margin[i] + debais_margin[i+batch_size]
            A_idx += 2
        elif race[i] == 2:
            C_num += debais_margin[i] + debais_margin[i+batch_size]
            C_idx += 2
        elif race[i] == 3:
            I_num += debais_margin[i] + debais_margin[i+batch_size]
            I_idx += 2
    if AA_idx == 0:
        AA_margin = 0
    else:
        AA_margin = AA_num / AA_idx
    if A_idx == 0:
        A_margin = 0
    else:
        A_margin = A_num / A_idx
    if C_idx == 0:
        C_margin = 0
    else:
        C_margin = C_num / C_idx
    if I_idx == 0:
        I_margin = 0
    else:
        I_margin = I_num / I_idx
    numerators = torch.exp((diagonal_cosine-debais_margin)/(beta))
    denominators=torch.sum(torch.exp(cosine_mat)*mask,dim=1)-torch.exp(diagonal_cosine/beta)+numerators # - x1x2/beta+ x1x2/beta+epsilon

    return -torch.mean(torch.log(numerators)-torch.log(denominators),dim=0), [AA_margin, A_margin, C_margin, I_margin]
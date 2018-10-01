import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt
import numpy as np

EPSILON = 1e-5

def get_pos_embeddings(positions, dim):
	embed = np.ones(len(positions), dim)*positions
	dims = np.arange()

def gen_weights(in_dim, out_dim):
	weight = init.uniform_(torch.empty(in_dim, out_dim))
	weight = Variable(weight, requires_grad=True)
	return weight

def gen_bias(out_dim):
	bias = torch.zeros(1, out_dim)
	bias = Variable(bias, requires_grad=True)
	return bias

class LayerNorm(nn.Module):

	def __init__(self, device):
		super(LayerNorm, self).__init__()
		self.gamma = torch.ones(1, dim, device=device)
		self.beta = torch.zeros(1, dim, device=device)
		self.gamma = Variable(self.gamma, requires_grad=True)
		self.beta = Variable(self.beta, requires_grad=True)

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True).sqrt()
		return self.gamma*(x - mu)/(sigma + EPSILON) + self.beta

class KQVAttention(nn.Module):

	def __init__(self, in_dim, out_dim, val_dim):
		super(KQVAttention, self).__init__()

		self.K_w = gen_weights(in_dim, out_dim)
		self.Q_w = gen_weights(in_dim, out_dim)
		self.V_w = gen_weights(in_dim, val_dim)
		self.out_dim = out_dim


	def forward(self, K, Q, V, mask): # TODO (ldery) : mask contains -inf in places where needs to be for encoder
		K = torch.matmul(K, self.K_w)
		Q = torch.matmul(Q, self.K_w)
		V = torch.matmul(V, self.K_w)

		logits = torch.matmul(Q, K.T) / sqrt(self.out_dim)
		logits = logits + mask
		embeddings = torch.matmul(nn.softmax(logits, dim=1), V)
		return embeddings

class FeedForward(nn.Module):

	def __init__(self, in_dim, inter_dim, out_dim):
		super(FeedForward, self).__init__()

		fc_1 = gen_weights(in_dim, inter_dim)
		b_1 = gen_bias(inter_dim)
		fc_2 = gen_weights(inter_dim, out_dim)
		b_2 = gen_bias(out_dim)

	def forward(self, x):
		h = torch.matmul(x, fc_1) + b_1
		h = F.relu(h)
		h = torch.matmul(h, fc_2) + b_2
		return h






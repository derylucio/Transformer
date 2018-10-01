import torch
import torch.nn as nn
from torch.autograd import Variable


class Transformer(nn.Module):

	def __init__(self):
		super(Transformer, self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()


class Encoder(nn.Module):

	def __init__(self):
		super(Encoder, self).__init__()


class Decoder(nn.Module):

	def __init__(self):
		super(Decoder, self).__init__()		
import json
import cv2
import numpy as np
import re, os, sys
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F

hidden_size = 300
embedd_size = 300
feature_depth = 512

epoch_num = 50
eval_interval =  1000
epoch_save_interval = 5
loss_average_interval = 1000
learning_rate = 0.001

SOS_token = 0	# start of sequence
EOS_token = 1	# end of sequence
MAX_LEN = 14 # max eval sequence length
lambda_param = 0.002 # attention regularization

model_path = './model'
train_images_dir_path = './train2014'
train_filepath_prefix = 'COCO_train2014_'
vgg19_path = './imagenet-vgg-verydeep-19.mat'
train_annotation_path = './annotations/captions_train2014.json'

vgg_layers = \
['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
'conv5_1', 'relu5_1', 'conv5_2']

class LanguageStats:
	
	def __init__(self):
		
		self.word2index = {"SOS": SOS_token, "EOS": 1}
		self.index2word = {SOS_token: "SOS", 1: "EOS"}
		self.words_num = 2

	def add_sentence(self, sentence):
		
		for word in sentence:
			self.add_word(word)

	def add_word(self, word):
		
		if word not in self.word2index:
			self.word2index[word] = self.words_num
			self.index2word[self.words_num] = word
            
			self.words_num += 1

def sent2seq(stats, sentence):
	
	return [stats.word2index[word] for word in sentence.split()]
	
def seq2sent(stats, sequence):
	
	return ' '.join([stats.index2word[indx] for indx in sequence]) 

def make_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory) 

def crop_center(img,crop):
	
    h, w, _ = img.shape
    startx = w/2 - (crop/2)
    starty = h/2 - (crop/2)    
      
    return img[starty:starty+crop, startx:startx+crop, :]

#-----------------------------------------------------------------------------------------------
#-------------------------------------- data reader --------------------------------------------
#-----------------------------------------------------------------------------------------------

class Reader(object):
	
	def __init__ (self, annotation_filepath, images_path, filepath_prefix):

		raw_data = open(annotation_filepath).read()
		samples = json.loads(raw_data)['annotations']
		
		self.captions = []
		self.filepaths = []
		
		for sample in samples:
			
			image_id = str(sample['image_id'])
			image_id = '0'*(12 - len(image_id)) + image_id # string completion to match MSCOCO namespace
			filepath = images_path + '/' + filepath_prefix+image_id + '.jpg'
			self.filepaths.append(filepath)
			
			caption = sample['caption']
			caption = re.sub('[!?.,"\']', '', caption)
			caption = caption.lower()
			self.captions.append(caption)
		
		self.lang_stats = LanguageStats()
		for caption in self.captions:
			self.lang_stats.add_sentence(caption.split())
		
		#self.captions = self.captions[:5000]
		#self.filepaths = self.filepaths[:5000]

		self.idx = 0
		self.samples_num = len(self.filepaths)
		self.choice = range(self.samples_num)
		np.random.shuffle(self.choice)
		
		print 'samples number:', self.samples_num
		print 'distinct words:', self.lang_stats.words_num
			
	def get_sample(self):
		
		while self.idx < self.samples_num:

			caption = self.captions[self.choice[self.idx]]
			sequence = sent2seq(self.lang_stats, caption)
			input_sequence = [SOS_token] + sequence
			target_sequence = sequence + [EOS_token]
			
			filepath = self.filepaths[self.choice[self.idx]]
			img = cv2.imread(filepath)
			
			self.idx = self.idx + 1
			
			yield img, input_sequence, target_sequence
		
		self.idx = 0
		np.random.shuffle(self.choice)	
	
	def prepare_image(self, img, shorter_dim_size=256, crop_size=224):
		
		h, w, _ = img.shape
		ratio = float(shorter_dim_size)/min(h, w)
		img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
		img = crop_center(img, crop_size)
		img = img.transpose(2, 0, 1)
		
		return img	
		
#-----------------------------------------------------------------------------------------------
#------------------------------------- vgg19 network -------------------------------------------
#-----------------------------------------------------------------------------------------------

class VGG19(nn.Module):
	
	def __init__ (self, path):
		
		super(VGG19, self).__init__()
		
		self.bgr_mean = Variable(torch.Tensor([103.939, 116.779, 123.68]).cuda()).view(1, 3, 1, 1)

		self.weights = []
		self.biases = []
		
		import scipy.io
		vgg_data = scipy.io.loadmat(path)['layers'][0]
		
		for i, vgg_layer in enumerate(vgg_layers):
			
			if 'conv' in vgg_layer:
				
				weight = vgg_data[i][0][0][0][0][0]
				bias = vgg_data[i][0][0][0][0][1][0]
								
				weight = weight.transpose(3, 2, 0, 1)
				
				weight = Variable(torch.from_numpy(weight).cuda(), requires_grad=False)
				bias = Variable(torch.from_numpy(bias).cuda(), requires_grad=False)
				
				self.weights.append(weight)
				self.biases.append(bias)
				
			else:
				
				self.weights.append(None)
				self.biases.append(None)

	def forward(self, x):
				
		x = x - self.bgr_mean
		
		for i, vgg_layer in enumerate(vgg_layers):
			
			if 'conv' in vgg_layer:
				
				w = self.weights[i]
				b = self.biases[i]
				x = torch.nn.functional.conv2d(x, weight=w, bias=b, stride=(1, 1), padding=(w.size()[2]/2, w.size()[3]/2))

			if 'relu' in vgg_layer:
				
				x = torch.nn.functional.relu(x)
				
			if 'pool' in vgg_layer:
				
				x = torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
			
		return x

#-----------------------------------------------------------------------------------------------
#--------------------------------------------- net ---------------------------------------------
#-----------------------------------------------------------------------------------------------

class Net(nn.Module):
	
	def __init__(self, hidden_size, embedd_size, vocab_size, feature_depth):
		
		super(Net, self).__init__()
        
		self.hidden_size = hidden_size
		self.embedd_size = embedd_size
		self.vocab_size = vocab_size
		self.feature_depth = feature_depth
		self.embedding = nn.Embedding(vocab_size, embedd_size)
 
		self.f_hidden_init = nn.Linear(feature_depth, hidden_size)
		self.f_memory_init = nn.Linear(feature_depth, hidden_size)
		
		self.f_attn = nn.Linear(feature_depth + hidden_size, 1)
		self.f_attn_gate = nn.Linear(hidden_size, 1)
		
		self.lstm = nn.LSTM(embedd_size + feature_depth, hidden_size)
		
		self.dropout = nn.Dropout(p=0.5)
		
		self.Ln = nn.Linear(hidden_size, embedd_size)
		self.Lz = nn.Linear(feature_depth, embedd_size)
		self.Lo = nn.Linear(embedd_size, vocab_size)

	def forward(self, features, input_sequence = None, mode='train'):
		
		features = features.view(1,feature_depth, -1) # (1, feature_depth, L)
		
		features_num = features.size()[2]

		average_feature = torch.mean(features, 2) # (1, 1, feature_depth)

		hidden_init = self.f_hidden_init(average_feature.unsqueeze(0))
		memory_init = self.f_memory_init(average_feature.unsqueeze(0))
		
		hidden = (hidden_init, memory_init)
				
		if mode == 'train':
		
			probs_sequence = []
			attns_sequence = []
			num_steps = input_sequence.size()[0]
			
			for idx in range(num_steps):
			
				input_token = input_sequence[idx]
				probs, hidden, attn = self.process_step(input_token, features, features_num, hidden, dropout='true')
					
				probs_sequence.append(probs.unsqueeze(0))
				attns_sequence.append(attn)
				
			probs_sequence = torch.cat(probs_sequence,0) # (seq_len, 1, vocab_size)
			attns_sequence = torch.cat(attns_sequence,0) # (seq_len, L, 1)

			return probs_sequence, attns_sequence
			
		elif mode == 'eval':
						
			input_token= Variable(torch.LongTensor([SOS_token]).cuda())
				
			output_sequence = []
			attn_sequence = []
			
			for idx in range(MAX_LEN):
				
				probs, hidden, attn = self.process_step(input_token, features, features_num, hidden)
				topv, topi = probs.topk(1)
				pred_token = topi[0]
				
				input_token = pred_token # for next iteration
				output_token = input_token[0].data.cpu().numpy()[0]
				
				if output_token == EOS_token:
					break
					
				attn_sequence.append(attn.data.cpu().numpy())
				output_sequence.append(output_token)		
			
			return output_sequence, attn_sequence
		
	def process_step(self, input_token, features,features_num, hidden, dropout='false'):
			
		prev_hidden = hidden[0]
				
		# prev_hiden - tensor of shape (1, 1, hidden_size)
		# features - tensor of shape (1, feature_depth, L)

		attn_input_prev_hidden = prev_hidden.expand(1, features_num, self.hidden_size) # (1, L, hidden_size)
		attn_input_features = features.transpose(1, 2)
		attn_input = torch.cat([attn_input_features, attn_input_prev_hidden], 2) # (1, L, hidden_size + feature_depth)
				
		attn = self.f_attn(attn_input)
		attn = F.softmax(attn, dim=1) # (1, L, 1)
		context_vector = torch.matmul(features.squeeze(0), attn.squeeze(0)) # (512, 1)

		attn_gate_input_prev_hidden = prev_hidden.squeeze(0) # (1, hidden_size)
		attn_gate = F.sigmoid(self.f_attn_gate(attn_gate_input_prev_hidden)) # (1, 1)			
		gated_context_vector = context_vector*attn_gate # (feature_depth, 1)
		gated_context_vector = gated_context_vector.transpose(0, 1) # (1, feature_depth)
				
		embedded = self.embedding(input_token) # (1, embedd_size)
				
		lstm_input = torch.cat([gated_context_vector, embedded], 1).unsqueeze(0) # (1, 1, feature_depth + embedd_size)
		_, hidden = self.lstm(lstm_input, hidden) # tuple, each element of shape (1, 1, hidden_size)
				
		next_hidden = hidden[0].squeeze(1) # (1, hidden_size)
		
		if dropout=='true':
			next_hidden=self.dropout(next_hidden)
		
		out = embedded + self.Ln(next_hidden) + self.Lz(gated_context_vector) # (1, embedd_size)
					
		probs = F.log_softmax(self.Lo(out), dim=1) # (1, vocab_size)
	
		return probs, hidden, attn

#------------------------------------------------------------------------------------------------------
#-------------------------------------------- training ------------------------------------------------
#------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
	
	vgg19 = VGG19(vgg19_path).cuda()
	reader = Reader(train_annotation_path, train_images_dir_path, train_filepath_prefix)
	vocab_size = reader.lang_stats.words_num
	net = Net(hidden_size, embedd_size, vocab_size, feature_depth).cuda()

	f_loss = nn.NLLLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

	make_dir(model_path)
	loss_averaged = 0.0

	for epoch in range(epoch_num):
		
		for iteration, (img, input_sequence, target_sequence) in enumerate(reader.get_sample()):
			
			img = reader.prepare_image(img)
			img = Variable(torch.from_numpy(img.astype(np.float32)).cuda())
			target_sequence = Variable(torch.LongTensor(target_sequence).cuda())
			teacher_sequence = Variable(torch.LongTensor(input_sequence).cuda())
			
			features = vgg19(img.unsqueeze(0)) # (1, feature_depth, H=14, W=14)
	
			probs_sequence, attns_sequence = net(features, teacher_sequence, mode='train')
			attns_sumed_over_sequence = torch.sum(attns_sequence,0).squeeze(1) # (L, )
			
			regularization = ((1.0 - attns_sumed_over_sequence)**2).sum()
			cross_entropy = f_loss(probs_sequence.squeeze(1), target_sequence)
			
			loss = cross_entropy + lambda_param*regularization
			
			optimizer.zero_grad()
			loss.backward()
			loss_averaged += loss.data[0]
			optimizer.step()

			if (iteration + 1) % eval_interval == 0:	

				output_sequence, attn_sequence = net(features, mode='eval')
				output_sentence = seq2sent(reader.lang_stats, output_sequence)
				target_sentence = seq2sent(reader.lang_stats, target_sequence.data.cpu().numpy()[:-1])
				
				print 'evaluation results:'
				print 'target_sentence:', target_sentence
				print 'output_sentence:', output_sentence
			
			if (iteration + 1) % loss_average_interval == 0:	
				
				print ('epoch:', epoch + 1, 'iter:', iteration + 1, 'loss:', loss_averaged/float(loss_average_interval))
				loss_averaged = 0.0		
				
		if (epoch + 1) % epoch_save_interval == 0:
			
			torch.save(net, model_path + '/net-epoch-' + str(epoch + 1) + '.pt')

	torch.save(net, model_path + '/net-final.pt')

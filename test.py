from train import *

test_img_path = './test/1.jpg'
net_path = './model/net-final.pt'
result_dir_path = './result'

vgg19 = VGG19(vgg19_path).cuda()	
net = torch.load(net_path).cuda()
train_reader = Reader(train_annotation_path, train_images_dir_path, train_filepath_prefix)

img = cv2.imread(test_img_path)
img = train_reader.prepare_image(img)
img_var = Variable(torch.from_numpy(img.astype(np.float32)).cuda())
features = vgg19(img_var.unsqueeze(0)) # (1, feature_depth, H=14, W=14)
output_sequence, attn_sequence = net(features, mode='eval')
output_sentence = seq2sent(train_reader.lang_stats, output_sequence)
print 'output sentence:', output_sentence

make_dir(result_dir_path)
img = img.transpose(1, 2, 0)
cv2.imwrite(result_dir_path + '/img.png', img)
h, w, _ = img.shape

for i, (output_word, attn_mat) in enumerate(zip(output_sentence.split(), attn_sequence)):

	attn_mat = attn_mat.reshape(14, 14)	
	attn_mat = cv2.resize(attn_mat, (h, w))
	attn_mat = cv2.GaussianBlur(attn_mat, (9, 9), 0)
	attn_mat = cv2.normalize(attn_mat, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	cv2.imwrite(result_dir_path + '/' + str(i) + '-' + output_word + '.png', (attn_mat*255).astype(np.uint8))

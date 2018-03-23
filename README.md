# Image Captioning

Generate images captions with attention mechanism and PyTorch.

VGG19 network is used to extract feature map from fourth convolutional layer before max pooling. At each timestep network produces probabilitiy distribution for next word in a sentence. Attentions are applied to each spatial location of feature map.

<img src="https://s9.postimg.org/g0zqqhyy7/image.png" width="320"> 

During generation of each word decoder can focus on certain portions of an image.

<img src="https://s14.postimg.org/pwhencbxd/canvas.png" width="420"> 

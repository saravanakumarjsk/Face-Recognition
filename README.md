# Face-Recognition
This repo is an implementation of MTCNN face recognition

## Requirements
```bash
pip install tensorflow=1.14.0
pip install mtcnn
pip install opencv-python
```

## Usage
Make sure you installed the requirements and cloned the repo.
````python
D:\face>python cam.py   # for web cam
D:\face>python image.py # for images
````

## Technique used
  It uses transfer learning to recognize faces from a video or image, the MTCNN is trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset and the weights are stored in and passed as default weights if no specefic weights file in given during the instaciation. These weights are then used to predict the faces in real time
  
## Loss function
  We use ````tf.nn.softmax_cross_entropy_with_logits ````because it minimizes the distance between two probability distributions ie., predicted and actual. Suppose the original image is of human, and model predicts [0.2, 0.7, 0.1] as probability for three classes where as true prob looks like [1,0,0], What we ideally want is that our predicted probabilites should be close to this original probability distribution. Here the probablity of predecting a face increases when the cross entropy loss decreases. This also helps the network to learn faster and avoid vanishing gradient problem when given along with softmax activation.

```python
softmax_loss = loss_weight[0] * \
    tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels[0],
                                            logits=cls_output))
````
## Accuracy
With default configurations the accuracy of the model is around 97% - 100%.
![alt text](https://github.com/saravanakumarjsk/Face-Recognition/blob/master/result2.jpg)
![alt text](https://github.com/saravanakumarjsk/Face-Recognition/blob/master/result.jpg)


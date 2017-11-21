[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project

In this project, I tried multiple different machines ranging from a Macbook Pro,
iMac, & MacPro. Even with the extra machine performance it was still a long time
to run each scenario. Also I decided to use the test data simply because I have
a time limit to pass. What I really enjoyed was learning about the different
types of neural network structures. At first I stuck with a baseline model and a
wide range of hyperparamters which will be covered later along with the types of
scenarios. However the second model seemed to come out better.

[image_0]: ./docs/misc/sim_screenshot.png
[network]: ./docs/misc/NeuralNetworkDiagram.png
[final-network]: ./docs/misc/FinalNeuralNetworkDiagram.png

![alt text][image_0]

### Key Functionality

The final neural network I built uses some key functions for it to work
correctly. Below is a description of each which, is required by the rubric.

###### 1. Encoders

```python
encoder_block(input_layer, filters, strides)
```

Encoders use separable convolution, which reduces the number of params needed,
increases efficiency, and reduces runtime performance.

This is a convolution layer. The purpose of this layer is to learn features with
spatial information.

In my project, the first encoder outputs a depth of 32 from 3 (input), and the
second encoder outputs a depth of 64.

The parameters for this layer are:

* **Input Layer:** The layer with the original image
* **Filter:** Information that is pulled from every pixel of the new layer
* **Strides:** Number of strides step we are doing when we slide the window. The
  default value is 1

###### 2. Decoders

```python
decoder_block(small_ip_layer, large_ip_layer, filters)
```

Decoders do bilinear sampling, which takes the input and first layer as inputs.
Skip connection will connect the output of one layer to a non-adjacent layer.

The inputs for this layer are:

* **Small IP Layer:** Input from the previous layer
* **Large IP Layer:** Skip layer
* **Filters:** Information that is pulled from every pixel of the new layer

###### 3. 1x1 Convolution Layers

This layer extracts information from the images before passing it onto the
decoder while it retains its spatial information.Another thing it does is reduce
the dimension of the layer.

This is a major difference between the fully connected layer, which would result
in the same amount of features. Also it requires the imaged to be flattened.

###### 4. Output Layers

This somewhat self explanatory it's the the final classification for each pixel
to the appropriate class. This is an another 1x1 convolution layer and has the 3
classifications classes. A softmax is used to the results in order to get a
proper probability.

### Neural Network Architecture

#### Model 1

After thinking about it for a bit, I decided on a multiple layered fully
convolutional neural network model. This model has 5 layers, 2 encoders, 2
decoders, and 1x1 convolution layer.

![Network Architecture][network]

##### Limitations

Unfortunately this didn't work well enough I consistently got a successful score
of 0.23-0.25. However I'll discuss this further in scenarios section.

#### Final Model

The second model I tried was very similar to the first model. However I added
two extra 1x1 convulsion layers after the input. After that the origin image is
concated on top of the previous results, which is very similar to a skip. The
goal being to use color channels to identify the mostly red hero.

![Network Architecture][final-network]

##### Limitations

This model could be better with more training images. Currently it has trouble
with the hero in the distance. Also it can misclassify the hero as other people.

The coolest thing about neural networks is they could identify other objects
(dog, cat, car, etc.). You simply need to provide an accurate model and record
enough training data.

### Hyperparameters

```python
learning_rate = 0.005
batch_size = 20
num_epochs = 10
steps_per_epoch = 250
validation_steps = 70
workers = 2
```

These were finally decided after some patience, mistakes, and errors. For
instance, resetting your training model after it ended without logging the
results. I tried as many hyperparamters as I could in the amount of time I had
left on this term.

Using "Final Model" with the above parameters the score I finally got was:
**0.422216381331**

This score was calculated with the following equation:

```python
average_IoU\*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative)).
```

#### Learning rate

After trying the different scenarios ranging from 0.001, 0.005, and 0.0005 I
came to determine that 0.005 gave somewhat better performance. However it didn't
impact training time as significantly.

#### Batch size and steps

This took a bit of playing around with trying to decide having the total
(batch_size \* steps) being over or under the total amount for the training
data. However I came to terms with a batch size of 20 and steps of 70. Although
increasing the steps increased training time the model percentage was much
better.

#### Epochs

This was also difficult and painful to test by trial and error. I tried 10 - 80.
However another over 40 tended to cause overfit or crash. I ended up using 10
because it seemed to decrease training time.

### Future Enhancements

I would have loved to put this on AWS EC2 environment but I'll test that when I
have more time. However there are a few things that could change the neural
network as well.

Also it would have been interesting to take a more detailed set of test data and
see if that helped with either of the models.

The last thing I can think of to tweak would have been the kernel size of the
encoders.

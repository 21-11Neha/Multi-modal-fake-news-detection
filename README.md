# Multi-modal-fake-news-detection
<br/>Performed binary classification for fake news detection
<br/>Extracted implicit and explicit features from both image and text using Bert and pretrained yolo model.
<br/>Applied multiheaded attention to effectively combine both image and text encodings and passed to a fully connected
layer for classification.



# Problem Statement
Given a set of m news articles containing the text and image information, we can
represent the data as a set of text-image tuples A = {( AT
 , AI
)} . In the fake news
detection problem, we want to predict whether the news articles in A are fake news
or not. We can represent the label set as Y = {0,1}, where 1 denotes real news
while 0 represents the fake news.

# Related Work 
Deception detection is a hot topic in the past few years. Deception information
includes scientific fraud, fake news, false tweets etc. Fake news detection is a
subtopic in this area. A TICNN based model has been proposed. In this model
besides the explicit features, it innovatively utilize two parallel CNNs to extract
latent features from both textual and visual information. And then explicit and
latent features are projected into the same feature space to form new
representations of texts and images. At last, it proposes to fuse textual and visual
representations together for fake news detection 
TI-CNN model consider both text and image information in fake news detection.
Beyond the explicit features extracted from the data, as the development of the
representative learning, convolutional neural networks are employed to learn the
latent features which cannot be captured by the explicit features. Finally, it utilizes
TI-CNN to combine the explicit and latent features of text and image information
into a unified feature space, and then use the learned features to identify the fake
news

# Proposed Model 
In this model we have divided the input layer into two branches viz text branch and
image branch . 
# Text branch: 
We have used Bert to convert text into vector form. After conversion
every word will be a vector of size 768. If a sentence contains n words then the
size of input vector formed will be nx768. As the number of words in different
sentences may vary, we have applied padding at the end of every vector
representation of sentences. 
# Image branch: 
We have used pre4trained yolo model which takes image as input
and outputs list of labels related to that image. We then have converted those list
into vector by passing those lists into bert. Again we have applied padding here to
make them equal sized vectors.
# Combining text and image branch: 
We have used multi-headed attention layer to
combine both image and text part. We have taken vector from text branch as key
and value. We have taken image as query for multiheaded attention layer. 
Output from the attention layer is padded, flattened and then passed to the fully
connected layer with input layer with (9984)13x768 neurons and output layer with
2 neurons. We have used softmax function in the output layer of fully connected
layer
<p align="center" width="100%">
    <img width="33%" src="https://user-images.githubusercontent.com/47311900/189345921-8ab4f8b9-3f65-4ba2-8a97-e3e84fdb3e0b.PNG">
</p>


# Results


![results](https://user-images.githubusercontent.com/47311900/189346874-10e67467-b9c5-4436-933a-b3c5906d02ba.PNG)


# Kernel image processing in NumPy (not finished)

In this post I will explain how I use NumPy to implement the **kernal image processing** that is used in Deep Learning and other image processing areas. After reading you will (hopefully) understand (better) how the convolutional in Neural Networks work, how **image bluring** like in Photoshop might work and how to implement that all in NumPy. To follow the post you need some basic knowledge of Python and NumPy.

## Intro

I'm currently watching a course about Deep Learning and reached the chapter about Convolutional Neural Networks (CNN) that are used to classify images. One of the most important features of the CNN are the **feature maps** that result from various filters applied for example at the beginning of a CNN directly on the classified image. As they can also **blur** or **sharpen** images in different ways I assume that they are also used in various software application.  

The strange looking black and white output in the image below is what CNN use to detect pattern for classification. I don't want to go into detail how exactly the CNN work, but you can be sure that there are more weird kernels.  

<div class="img-center">

<img width="400px" src="/images/kernel_original_outputs.png" alt="Orignal with Outputs">

Source: Wikipedia - Kernel (image processing)
</div>

## What is a convolutional kernel?

To understand what a convolutional kernel is you need to understand what an image actually is. Well, or how a computer sees an image. You know that for a computer an image is simply a rectangle of pixels. Pixels can be expressed as numbers. An rectangle of numbers is a **matrix**, right?

The next image illustrates on the left an image. Every pixel has it's own number. The **kernel** is also a matrix, but way smaller. You probably wonder where size of the matrix and it's numbers are coming from. Depending what you want to achieve with your kernel - sharpening, blurring or detecting edges - you need to choose a special combination of numbers that fit to your goal. Now you probably wonder how just a few numbers can blur an image. Don't worry you will understand it in a few minutes.  

<div class="img-center">

<img width="400px" src="/images/convolve.png" alt="Convolution">

Source: MINES ParisTech
</div>

Now let's look at the last step in the image. How do I get the *4*. The idea is very simple. You put the kernel on the image and multiply the elements of the kernel and the part of the image with each other. Don't use matrix multiplication. Use the simple multiplication (there is probably a term for it). If you use the data in the image you will get this matrix.  

$$
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
1 & 0 & 1
\end{bmatrix}
$$  

Then you sum up all values, here 1+1+1+1 and put it in a seperate **result matrix**. And how do I get all the other values. Well, you just move the kernel over the image. Well how do I move the kernel over the image. The are different ways and some limitation. To simplify the idea, just put the kernel in the top left corner for the start and then move it to the right. How much you move to the right is determined by the number of **strides**, let's set this number to one. Once you reach the end of the image you move your kernel down vertically and then start again on the left edge. You do this until you reach the bottom right corner.  

If you look at the image and imagine how the kernel is moving over the image you can visualize that the kernel perfectly covers all pixel. But what if the number of strides is not one but two. The kernel will move to a position where it's numbers don't overlap with the image number. In this case you need a **padding**. A padding is like a another row or column in your input matrix. There are different ways to implement a padding. In this post I will not cover it as it is not neccesary to understand the idea behind the kernel processing.

## Let's code

Let us assume that we don't know how to implement padding, or we just are lazy and don't want to have it. How can I know that the combination of my kernel and the strides fit to the image, so that we don't need a padding. There is a formula that can help us.

$$
Output_x = \frac{Input_x + 2\cdot Padding_x - Kernel_x}{strides} + 1
$$

If your image is 100 pixel wide, your kernel has a width of 3 and the strides are set to 1, the width of your output will be 98. This number is not a floating number but an integer. This means that you don't need a padding in form of columns. You can do the same for the vertical length. If one of the output dimensions is a floating number you need some padding. Technically you could also cut your image so that it fits, but this sounds like cheating. You may also notice that the output is smaller than the input. This means that we will use some information, right? That's correct but absolutely not a problem for a CNN. Why? That's complicated and not neccesary here.

Ok, let's implement this in Python. The output of the code cells is written in the comments indicated by `#`.
The only libraries that you need here are

```python
import numpy as np

from matplotlib.image import imread
import matplotlib.pyplot as plt
```
With `matplotlib` we will read images locally or from the internet and then show them.
```python
animal = imread('https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png')
plt.imshow(animal)
```

<img src="/images/output1.png">

`imread(...)` outputs an NumPy array that contains all pixels of the image.   
How big is this image?
```python
animal.shape
# (100, 100, 3)
```
The output tells us that the image is 100 pixel wide, 100 pixel long. The 3 shows us the **3rd** dimension - the color dimension. The image is not a black/white image. Each pixel contains information about the three color channels **red**, **green**, **blue**. For a black/white images the output of the shape would be simply `(100, 100)`.  
We can also look at each pixel seperately. This is the pixel in the top left corner.
```python
animal[0, 0, :3]
# array([0.5176471, 0.5137255, 0.5372549], dtype=float32)
```
The colors in the channel can be expressed either on the scale from 0 to 255, where you can only use integers, or 0 to 1, where you can obviously also use floating numbers.

Our kernels are also NumPy arrays.
```python
kernel_blur = np.ones((3,3))*(1/9)
kernel_blur
# array([[0.11111111, 0.11111111, 0.11111111],
#       [0.11111111, 0.11111111, 0.11111111],
#       [0.11111111, 0.11111111, 0.11111111]])
```
This might help you to understand how a kernel can blur a image. Every single pixel in the final output will be a combination of 9 diffent pixel surrounding the

## Trash


```python

```
```

```
$$
\begin{bmatrix}
4 & 0 & 0\\
0 & 0 & 0\\
0 & 0 & -4
\end{bmatrix}
$$

## Sources

<a href="http://perso.mines-paristech.fr/fabien.moutarde/ES_MachineLearning/TP_convNets/convnet-notebook.html" target="_new">MINES - ParisTech</a>

<a href="https://en.wikipedia.org/wiki/Kernel_(image_processing)" target="_new">Wikipedia - Kernel-(image processing)  
</a>

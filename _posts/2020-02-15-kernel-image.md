# Kernel image processing in NumPy

In this post I will explain how I use NumPy to implement the **kernal image processing** that is used in Deep Learning and other image processing areas.

## Intro

I'm currently watching a course about Deep Learning and reached the chapter about Convolutional Neural Networks (CNN) that are used to classify images. One of the most important features of the CNN are the **feature maps** that result from various filters applied for example at the beginning of a CNN directly on the classified image. As they can also **blur** or **sharpen** images in different ways I assume that they are also used in various software application.  

The strange looking black and white output in the image below is what CNN use to detect pattern for classification. I don't want to go into detail how exactly the CNN work, but you can be sure that there are more weird kernels.  

<div align="center">

<img width="400px" src="/images/kernel_original_outputs.png" alt="Orignal with Outputs">

Wikipedia - Kernel (image processing)
</div>

## What is a convolutional kernel?

To understand what a convolutional kernel is you need to understand what an image actually is. Well, or how a computer sees an image. You know that for a computer an image is simply a rectangle of pixels. Pixels can be expressed as numbers. An rectangle of numbers is a **matrix**, right?

The next image illustrates on the left an image. Every pixel has it's own number. The **kernel** is also a matrix, but way smaller. Your probably wonder where size of the matrix it's numbers are coming from. Depending what you want to achieve with your kernel - sharpening, blurring or detecting edges - you need to chose a special combination of numbers that fit to your goal. Now you probably wonder how just a few numbers can blur an image. Don't worry you will understand it in a few minutes, or not.  

<div align="center">

<img width="400px" src="/images/convolve.png" alt="Convolution">

MINES ParisTech
</div>

Now let's look at the last step in the image. How do I get the *4*. The idea is very simple. You put the kernel on the image and multiply the elements of the kernel and the part of the image with each other. Don't use matrix multiplication. Use the simple multiplication (there is probably a term for it). If you the data in the used image you will get this matrix.  

$$
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
1 & 0 & 0
\end{bmatrix}
$$  

Then you sum up all values, here 1+1+1+1 and put it in a seperta **result matrix**. And how do I get all the other values. Well, you just move the kernel over the image. Well how do I move the kernel over the image. The are different ways and some limitation. To simplify the idea, just put the kernel in the top left corner for the start and then move it to the right. How much you move to the right is determined by the number of **strides**, let's set this number to one. Once you reach the end of the image you move your kernel down vertically and the start again on the left edge. You do this until you reach the bottom right corner.  

If you look at the image and imagine how the kernel is moving over the image you can visualize that the kernel perfectly covers all pixel. But what if the number of strides is not one but two. The kernel will move to a position where it's number don't overlap with the image. In this case you need a **padding**. There are different ways to implement a padding. In this post I will not cover it as it is not neccesary to understand the idea behind the kernel processing.

## Let's code

Let us assume that we don't know how to implement padding, or we just are lazy and don't want to have it. How can I know that the combination of my kernel and the strides fit to the image, so that we don't need a padding. There is a formula that can help us.

$$
Output_x = \frac{Input_x + 2\cdotPadding_x - Kernel_x}{strides} + 1
$$
If your image is 100 pixel wide, your kernel has a width of 3 and the strides are set to 1, the width of your output will be 98. This number is not a floating number but an integer. This means that you don't need a horizontal padding. You can do the same for the vertical length. If one of the output dimensions is a floating number you need some padding. Technically you could also cut your image so that it fits, but this sounds like cheating. You may also notice that the output is smaller than the input. This means that we will use some information, right? That's correct but alsolutely not a problem for a CNN. Why? That's complicated and not neccesary here.

Ok, let's implement this in python. (btw I use python 3.7)
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

MINES - ParisTech  
http://perso.mines-paristech.fr/fabien.moutarde/ES_MachineLearning/TP_convNets/convnet-notebook.html

Wikipedia - Kernel-(image processing)  
https://en.wikipedia.org/wiki/Kernel_(image_processing)
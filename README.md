## STAT 5720 Project 2: Reproducing Results from "Deep Neural Networks as Gaussian Processes"

Added by E. Noonan

*Note: This has been appended to the original README from the "NNGP: Deep Neural Network Kernel for Gaussian Process"
repo README. The original README content remains at the end of this document for historical and reference purposes.*

### Task Summary
The objective is to create a dockerfile that can run the repo code from the 
source paper, to add code to recreate at least one figure from the paper, and
to extend the code to conduct additional analysis not included in the original work.

Option 2 was selected and the assignment description is included below for reference.
#### Option 2: Predictive uncertainty vs. prediction error
Reproduce the paper's finding (Figure 3) that the GP's predictive variance correlates 
with its actual squared prediction error. This is directly computable from the GP 
regression outputs the existing code already produces internally. You will need to 
extract and plot per-example predictive variance against per-example squared error.

### Results Summary
The instructor-provided script, uncertainty_plot.py, recreates Figure 3 from the paper. 

![](C:\Users\eenoo\Pictures\Screenshots\NNGP_Figure_3.png)
>**Figure 1.** Prediction uncertainty for MNIST and CIFAR-10 datasets 
>(Figure 3 from "Deep Neural Networks as Gaussian Processes")

The recreation of the plot used a reduced training set of 1000 data points instead of 
the 45k used for CIFAR-10 and 50K used for MNIST in NNGP Figure 3 (Figure 1 above). This reduced training set
reduced the processing time required to generate the plot. Despite this difference in 
data volume, the reproduced plot shows the same general clustering and relationships for
both nonlinearities. Additionally, NNGP Figure 8 in the appendix (Figure 2 below) includes these same plots with 
reduced number of training points (1000 and 5000). The calculated correlation values are summarized in Table 1. 

![](C:\Users\eenoo\Pictures\Screenshots\NNGP_Figure_8.png)

> **Figure 2.** Prediction uncertainty for smaller number of training points 
> (Figure 8 from "Deep Neural Networks as Gaussian Processes")

> **Table 1.** Comparison of Correlation Values for NNGP Paper and Reproduction Code 
> 
| Data Set     | Training Set | Tanh Corr |              | ReLU Corr  |              |
|--------------|--------------|-----------|--------------|------------|--------------|
|              |              | *Paper*   | *Recreation* | *Paper*    | *Recreation* | 
| **MNIST**    | 50k          | 0.9330    | -            | **0.9573** | -            |
|              | 5k           | 0.9583    | **0.9725**   | 0.9701     | 0.9710       |
|              | 1k           | 0.9792    | 0.9833       | 0.9806     | **0.9840**   |
| **CIFAR-10** | 45k          | 0.7428    | -            | **0.8223** | -            |
|              | 5k           | 0.8176    | **0.8636**   | 0.8411     | 0.7948       |
|              | 1k           | 0.8851    | **0.9366**   | 0.8430     | 0.7200       |

The code recreating the paper images produced plots that show similar trends and generally calculated nonlinearity 
correlation values on the same order as the paper's though it is noted that not all correlation values had the 
same best performer as the paper.

To extend this analysis, additional datasets were selected to use for evaluation of the NNGP approach and the 
training set size was extended to include an option of 5000 in addition to the 1000 option. Two data sets were selected 
for this task: Kuzushiji-MNIST and Fashion MNIST. These data sets were identified as similar to but more complex than 
the MNIST data set. As both additional data sets are gray scale like MNIST and CIFAR-10 uses color images, the 
expectation was that these data sets would fall between the MNIST and CIFAR-10 in complexity.

The Kuzushiji-MNIST dataset consists of 70,000 28x28 grayscale images of Japanese Hiragana characters 
that can be used as a drop in replacement for MNIST but is considered a more challenging dataset than the baseline 
MNIST. Citations for this dataset are: 
"KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
and the associated paper: Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718
The dataset and additional details are available on GitHub: https://github.com/rois-codh/kmnist.

The Fashion MNIST dataset is a set of 70,000 28x28 grayscale images of clothing items taken from Zalando article images. 
The dataset is intended to be used as a drop in replacement for MNIST but is considered a more complex image dataset. 
The dataset and additional details are available on GitHub: https://github.com/zalandoresearch/fashion-mnist.

![](C:\Users\eenoo\Downloads\STAT_5720_Projects\Proj_NNGP - Copy\NNGP_Extension_Graphical_Results.png)
> **Figure 3.** Prediction uncertainty from Extended Analysis

> **Table 2.** Comparison of Correlation Values from Extended Analysis 
>
| Data Set | 1k Tanh Corr | 5k Tanh Corr | 1k ReLU Corr | 5k ReLU Corr |
|----------|--------------|--------------|--------------|--------------|
| MNIST    | 0.9833       | 0.9725       | 0.9840       | 0.9710       |
| KMNIST   | 0.9884       | 0.9372       | 0.9908       | 0.9555       |
| FMNIST   | 0.9573       | 0.9693       | 0.9671       | 0.9665       |
| CIFAR-10 | 0.9366       | 0.8636       | 0.7200       | 0.7948       | 

The additional datasets show similar correlations in the MSE output variation relationships and, as anticipated, the 
graphical clustering of the results show a progression that suggests that the complexity of the images increases the 
scatter within the data points. The MNIST and KMNIST, which are single character images show cleaner separation between
the two nonlinearities. The Fashion MNIST and CIFAR-10 data sets are more complex graphical images, with the addition of
color to the CIFAR-10 images further increasing complexity. The outcomes from these datasets show more scatter as well 
as more overlap between the groups. The CIFAR-10 has the lowest correlation scores, which is not unexpected for the 
most complex dataset. For KMNIST and Fashion MNIST, the higher correlation values switched depending on the training 
size, with the 1k sets favoring KMNIST and the 5K sets favoring Fashion MNIST.  





### Execution Instructions

In order to copy and run this code, follow the steps below:

1. git clone https://github.com/eenoonan/nngp_project
2. cd nngp_project
3. docker build --platform linux/amd64 -t nngp-project .
4. docker run --platform linux/amd64 -it -v "$(pwd)/output:/nngp/output" nngp-project
5. 

### Limitations

This effort does not attempt to recreate the full training set size runs that are used to generate Figure 3 in the 
source paper. Those were 45k and 50k training sets and would have required significant computational time and power
beyond the scope of this exercise. Rather, 1k and 5k training sets were used to approximate the resulting plots and to 
generate similar ones for the new data sets. 

_____________________________________

# NNGP: Deep Neural Network Kernel for Gaussian Process

TensorFlow open source implementation of

[**Deep Neural Networks as Gaussian Processes**](https://arxiv.org/abs/1711.00165)


by Jaehoon Lee*, Yasaman Bahri*, Roman Novak, Sam Schoenholz, Jeffrey Pennington,
Jascha Sohl-dickstein

Presented at the International Conference on Learning Representation(ICLR) 2018.

## UPDATE (September 2020):
See also [Neural Tangents: Fast and Easy Infinite Neural Networks in Python](https://arxiv.org/abs/1912.02803) (ICLR 2020)
available at [github.com/google/neural-tangents](https://github.com/google/neural-tangents) for 
more up-to-date progress on computing NNGP as well as NT kernels supporting wide variety of architectural components.


## Overview
A deep neural network with i.i.d. priors over its parameters is equivalent to a 
Gaussian process in the limit of infinite network width. The Neural Network
Gaussian Process (NNGP) is fully described by a covariance kernel determined by 
corresponding architecture.

This code constructs covariance kernel for the Gaussian process that is equivalent to
infinitely wide, fully connected, deep neural networks. 

## Usage

To use the code, run `run_experiments.py`,
which uses NNGP kernel to make full Bayesian prediction on the MNIST dataset.


```python
python run_experiments.py \
       --num_train=100 \
       --num_eval=10000 \
       --hparams='nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83' \
```

## Contact
***Code author:*** Jaehoon Lee, Yasaman Bahri, Roman Novak

***Pull requests and issues:*** @jaehlee

## Citation
If you use this code, please cite our paper:
```
  @article{
    lee2018deep,
    title={Deep Neural Networks as Gaussian Processes},
    author={Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington, Jascha Sohl-dickstein},
    journal={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=B1EA-M-0Z},
  }
```

## Note

This is not an official Google product.

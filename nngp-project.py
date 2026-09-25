# nngp-project.py

# This script executes the necessary commands to download, process, and plot datasets
# for a Neural Network Gaussian Process kernel demonstration that is based on the work of
# the authors of the below paper and the associated GitHub repo
#
# "Deep Neural Networks as Gaussian Processes" by
# Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz,
# Jeffrey Pennington, Jascha Sohl-Dickstein
# arXiv:1711.00165 (https://arxiv.org/abs/1711.00165).
#
# The work in this script expands upon the baseline work by incorporating code to recreate
# Figures 3 and 8 of the paper for the MNIST and CIFAR-10 datasets (leveraging code (uncertainty_plot.py) provided
# by the instructor for STAT 5720 at University of Colorado Boulder Fall 2026).
# The script uncertainty_plot.py was further modified (uncertainty_plot_updated.py) as part of this exercise
# to do the following:
# 1. Add the option to use a training size of 5000 as well as the existing size of 1000
# 2. Incorporate options to evaluate additional datasets Kuzushiji-MNIST (kmnist) and Fashion MNIST (fmnist)
# 3. Solicit user input to run the 1000, 5000 or both for the four available data sets

# The Kuzushiji-MNIST dataset consists of 70,000 28x28 grayscale images of Japanese Hiragana characters
# that can be used as a drop in replacement for MNIST but is considered a more challenging dataset than the baseline MNIST.
# Citations for this dataset are:
# "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341
# and the associated paper: Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. arXiv:1812.01718
# The dataset and additional details are available on GitHub: https://github.com/rois-codh/kmnist.

# The Fashion MNIST dataset is a set of 70,000 28x28 grayscale images of clothing items taken from Zalando article images.
# The dataset is intended to be used as a drop in replacement for MNIST but is considered a more complex image dataset.
# The dataset and additional details are available on GitHub: https://github.com/zalandoresearch/fashion-mnist.

import subprocess

configs_all = [
    # 1000 point runs for mnist, cifar10, kmnist, and fmnist
    {
        'dataset': 'mnist',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_mnist_1k.png',
    },
    {
        'dataset': 'cifar10',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_cifar10_1k.png',
    },
    {
        'dataset': 'kmnist',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_kmnist_1k.png',
    },
    {
        'dataset': 'fmnist',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_fmnist_1k.png',
    },
    # 5000 point runs for mnist, cifar10, kmnist, and fmnist
    {
        'dataset': 'mnist',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_mnist_5k.png',
    },
    {
        'dataset': 'cifar10',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_cifar10_5k.png',
    },
    {
        'dataset': 'kmnist',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_kmnist_5k.png',
    },
    {
        'dataset': 'fmnist',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_fmnist_5k.png',
    }
]

configs_1k = [
    # 1000 point runs for mnist, cifar10, kmnist, and fmnist
    {
        'dataset': 'mnist',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_mnist_1k.png',
    },
    {
        'dataset': 'cifar10',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_cifar10_1k.png',
    },
    {
        'dataset': 'kmnist',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_kmnist_1k.png',
    },
    {
        'dataset': 'fmnist',
        'num_train': 1000,
        'num_eval': 1000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_fmnist_1k.png',
    }
]
configs_5k = [
    # 5000 point runs for mnist, cifar10, kmnist, and fmnist
    {
        'dataset': 'mnist',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_mnist_5k.png',
    },
    {
        'dataset': 'cifar10',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_cifar10_5k.png',
    },
    {
        'dataset': 'kmnist',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_kmnist_5k.png',
    },
    {
        'dataset': 'fmnist',
        'num_train': 5000,
        'num_eval': 5000,
        'params': 'depth=3,weight_var=2.0,bias_var=0.2',
        'nonlinearities': 'tanh,relu',
        'output_file': '/nngp/output/uncertainty_fig3_fmnist_5k.png',
    }
]

# prompt user to select training size to run
configs_select = input("Select the training set size to run: 1. 1k only, 2. 5k only, 3. All (1k & 5k) ")

# set training size configs based on user input
if configs_select == '1':
    configs = configs_1k
elif configs_select == '2':
    configs = configs_5k
else:
    configs = configs_all

# pass selected configs to run code and generate Fig 3 for all datasets
for i, cfg in enumerate(configs):
    args = ['python', 'uncertainty_plot_updated.py'] + [f'--{k}={v}' for k, v in cfg.items()]
    print(f"[{i+1}/{len(configs)}] Running: {' '.join(args)}")
    result = subprocess.run(args) #capture_output=True, text=True
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr}")
    else:
        print(f"  Done -> {cfg['output_file']}")



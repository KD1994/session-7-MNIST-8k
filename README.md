<h1 align="center">
  Simple Neural Network for MNIST
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3810/">
    <img src="https://img.shields.io/badge/python-3.8-blue.svg" alt="Python 3.8">
  </a>
  <a href="Static Badge">
    <img src="https://img.shields.io/badge/PyTorch-1.13.1-red" alt="PyTorch-1.13.1"/>
  </a>
  <a href="Static Badge">
    <img src="https://img.shields.io/badge/Acc-99.4%25-green" alt="Acc-99.4"/>
  </a>
</p>

<p align="center">
Train a neural network to classify MNIST digits with less than 8K parameters with test accuracy more than 99.4% within 15 epochs and describe the model architecture and training analysis.
</p>


# Requirements

- Python 3.8
- PyTorch
- torchvision
- torchinfo


# [MnistNet_1](MNIST_S7_T1.ipynb)

## Target
- Try to reduce parameters to near 8K with more than 98% test accuracy within 10 epochs
- Try without Batch normalization and regularization

## Result
- Parameters: 8,620
- Best Training Accuracy: 98.66%
- Best Test Accuracy: 98.37%

## Analysis
- Ignoring first few epochs, before 10 epochs, the model looks like underfitting.
- After 10 epochs, the train & test gap is getting narrower and training accuracy is getting higher compared to test accuracy.
- Need to reduce parameters to less than 8k and improve model skeleton.


## Network Architecture 

| INPUT    | KERNEL     | OUTPUT |
|----------|------------|--------|
| 28x28x1  | 3x3x1x8    | 26     |
| 26x26x8  | 3x3x8x16   | 24     |
| 24x24x16 | 3x3x16x16  | 22     |
| 22x22x16 | 1x1x16x16  | 22     |
| Maxpool()|            | 11     |
| 11x11x8  | 3x3x8x8    | 9      |
| 9x9x8    | 3x3x8x8    | 7      |
| 7x7x8    | 3x3x8x16   | 5      |
| 7x7x16   | 1x1x16x10  | 5      |
| 5x5x10   | 5x5x10x10  | 1      |
| Flatten()             |        | 


## RF Calculation

**Formula of RF**: \(r_{out}\) = \(r_{in}\) + (\(k\) - 1) * \(j_{in}\)

**Formula of Jump**: \(j_{out}\) = \(j_{in}\) * \(s\)


| \(n_{in}\) | \(k\) | \(s\) | \(p\) | \(n_{out}\) | \(r_{in}\) | \(j_{in}\) | \(r_{out}\) | \(j_{out}\) |
|------------|-------|-------|-------|-------------|------------|------------|-------------|-------------|
| 28         | 3     | 1     | 0     | 26          | 1          | 1          | 3           | 1           |
| 26         | 3     | 1     | 0     | 24          | 3          | 1          | 5           | 1           |
| 24         | 3     | 1     | 0     | 22          | 5          | 1          | 7           | 1           |
| 22         | 1     | 1     | 0     | 22          | 7          | 1          | 7           | 1           |
| 22         | 2     | 2     | 0     | 11          | 7          | 1          | 8           | 2           |
| 11         | 3     | 1     | 0     | 9           | 8          | 2          | 12          | 2           |
| 9          | 3     | 1     | 0     | 7           | 12         | 2          | 16          | 2           |
| 7          | 3     | 1     | 0     | 5           | 16         | 2          | 20          | 2           |
| 5          | 5     | 1     | 0     | 1           | 20         | 2          | 28          | 2           |


## Model Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MnistNet_1                               [1, 10]                   --
├─Sequential: 1-1                        [1, 8, 11, 11]            --
│    └─Conv2d: 2-1                       [1, 8, 26, 26]            72
│    └─ReLU: 2-2                         [1, 8, 26, 26]            --
│    └─Conv2d: 2-3                       [1, 16, 24, 24]           1,152
│    └─ReLU: 2-4                         [1, 16, 24, 24]           --
│    └─Conv2d: 2-5                       [1, 16, 22, 22]           2,304
│    └─ReLU: 2-6                         [1, 16, 22, 22]           --
│    └─Conv2d: 2-7                       [1, 8, 22, 22]            128
│    └─MaxPool2d: 2-8                    [1, 8, 11, 11]            --
├─Sequential: 1-2                        [1, 10, 5, 5]             --
│    └─Conv2d: 2-9                       [1, 8, 9, 9]              576
│    └─ReLU: 2-10                        [1, 8, 9, 9]              --
│    └─Conv2d: 2-11                      [1, 8, 7, 7]              576
│    └─ReLU: 2-12                        [1, 8, 7, 7]              --
│    └─Conv2d: 2-13                      [1, 16, 5, 5]             1,152
│    └─ReLU: 2-14                        [1, 16, 5, 5]             --
│    └─Conv2d: 2-15                      [1, 10, 5, 5]             160
│    └─ReLU: 2-16                        [1, 10, 5, 5]             --
├─Sequential: 1-3                        [1, 10, 1, 1]             --
│    └─Conv2d: 2-17                      [1, 10, 1, 1]             2,500
==========================================================================================
Total params: 8,620
Trainable params: 8,620
Non-trainable params: 0
Total mult-adds (M): 2.00
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.22
Params size (MB): 0.03
Estimated Total Size (MB): 0.26
==========================================================================================
```

## Data Transformations

```python
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
```


## Training Logs

```
Epoch=0 Batch=468 loss=2.3021410 Accuracy=14.03%: 100%|██████████| 469/469 [00:05<00:00, 79.93it/s]

Test set: Average loss: 0.0181878, Accuracy: 1682/10000 (16.82%)

Epoch=1 Batch=468 loss=0.1666424 Accuracy=45.83%: 100%|██████████| 469/469 [00:06<00:00, 76.49it/s]

Test set: Average loss: 0.0028808, Accuracy: 8915/10000 (89.15%)

Epoch=2 Batch=468 loss=0.0895845 Accuracy=93.52%: 100%|██████████| 469/469 [00:06<00:00, 77.77it/s]

Test set: Average loss: 0.0011953, Accuracy: 9538/10000 (95.38%)

Epoch=3 Batch=468 loss=0.1695518 Accuracy=95.92%: 100%|██████████| 469/469 [00:06<00:00, 76.89it/s]

Test set: Average loss: 0.0008362, Accuracy: 9658/10000 (96.58%)

Epoch=4 Batch=468 loss=0.1363637 Accuracy=96.81%: 100%|██████████| 469/469 [00:06<00:00, 77.89it/s]

Test set: Average loss: 0.0006801, Accuracy: 9724/10000 (97.24%)

Epoch=5 Batch=468 loss=0.1017923 Accuracy=97.25%: 100%|██████████| 469/469 [00:06<00:00, 74.53it/s]

Test set: Average loss: 0.0006071, Accuracy: 9768/10000 (97.68%)

Epoch=6 Batch=468 loss=0.0555718 Accuracy=97.55%: 100%|██████████| 469/469 [00:06<00:00, 76.80it/s]

Test set: Average loss: 0.0004886, Accuracy: 9804/10000 (98.04%)

Epoch=7 Batch=468 loss=0.1792802 Accuracy=97.81%: 100%|██████████| 469/469 [00:06<00:00, 77.53it/s]

Test set: Average loss: 0.0005028, Accuracy: 9813/10000 (98.13%)

Epoch=8 Batch=468 loss=0.0855415 Accuracy=97.98%: 100%|██████████| 469/469 [00:06<00:00, 74.57it/s]

Test set: Average loss: 0.0004819, Accuracy: 9801/10000 (98.01%)

Epoch=9 Batch=468 loss=0.0208282 Accuracy=98.07%: 100%|██████████| 469/469 [00:06<00:00, 78.14it/s]

Test set: Average loss: 0.0003952, Accuracy: 9837/10000 (98.37%)

Epoch=10 Batch=468 loss=0.0225720 Accuracy=98.20%: 100%|██████████| 469/469 [00:05<00:00, 78.25it/s]

Test set: Average loss: 0.0004437, Accuracy: 9834/10000 (98.34%)

Epoch=11 Batch=468 loss=0.0932789 Accuracy=98.34%: 100%|██████████| 469/469 [00:05<00:00, 79.44it/s]

Test set: Average loss: 0.0004042, Accuracy: 9833/10000 (98.33%)

Epoch=12 Batch=468 loss=0.0520834 Accuracy=98.49%: 100%|██████████| 469/469 [00:06<00:00, 74.45it/s]

Test set: Average loss: 0.0003858, Accuracy: 9834/10000 (98.34%)

Epoch=13 Batch=468 loss=0.0801769 Accuracy=98.42%: 100%|██████████| 469/469 [00:06<00:00, 74.50it/s]

Test set: Average loss: 0.0004621, Accuracy: 9811/10000 (98.11%)

Epoch=14 Batch=468 loss=0.0116184 Accuracy=98.66%: 100%|██████████| 469/469 [00:06<00:00, 74.69it/s]

Test set: Average loss: 0.0004304, Accuracy: 9824/10000 (98.24%)
```


# [MnistNet_2](MNIST_S7_T2.ipynb)

## Target
- Parameters < 8K
- Batch normalization and regularization
- play around with dropout rate

## Result
- Parameters: 7,416
- Best Training Accuracy: 98.87%
- Best Test Accuracy: 99.47%

## Analysis
- Test accuracy reached 99.47% but consistency is not good enough
- Train-test gap can be seen as train accuracy doesn't reach 99%
- Can try to reduce parameters futher below 7k and tweak around dropout rate (0.05) a bit more


## Network Architecture 

| INPUT    | KERNEL     | OUTPUT |
|----------|------------|--------|
| 28x28x1  | 3x3x1x8    | 26     |
| 26x26x8  | 3x3x8x16   | 24     |
| 24x24x16 | 1x1x16x8   | 24     |
| Maxpool()|            | 12     |
| 12x12x8  | 3x3x8x16   | 10     |
| 10x10x16 | 3x3x16x16  | 8      |
| 8x8x16   | 3x3x16x16  | 6      |
| AdaptiveAvgPool()|    | 1      | 
| 1x1x16   | 1x1x16x10  | 1      |
| Flatten()             |        | 


## RF Calculation

**Formula of RF**: \(r_{out}\) = \(r_{in}\) + (\(k\) - 1) * \(j_{in}\)

**Formula of Jump**: \(j_{out}\) = \(j_{in}\) * \(s\)


| \(n_{in}\) | \(k\) | \(s\) | \(p\) | \(n_{out}\) | \(r_{in}\) | \(j_{in}\) | \(r_{out}\) | \(j_{out}\) |
|------------|-------|-------|-------|-------------|------------|------------|-------------|-------------|
| 28         | 3     | 1     | 0     | 26          | 1          | 1          | 3           | 1           |
| 26         | 3     | 1     | 0     | 24          | 3          | 1          | 5           | 1           |
| 24         | 1     | 1     | 0     | 24          | 5          | 1          | 5           | 1           |
| 24         | 2     | 2     | 0     | 12          | 5          | 1          | 6           | 2           |
| 12         | 3     | 1     | 0     | 10          | 6          | 2          | 10          | 2           |
| 10         | 3     | 1     | 0     | 8           | 10         | 2          | 14          | 2           |
| 8          | 3     | 1     | 0     | 6           | 14         | 2          | 18          | 2           |


## Model Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MnistNet_4                               [1, 10]                   --
├─Sequential: 1-1                        [1, 8, 12, 12]            --
│    └─Conv2d: 2-1                       [1, 8, 26, 26]            72
│    └─ReLU: 2-2                         [1, 8, 26, 26]            --
│    └─BatchNorm2d: 2-3                  [1, 8, 26, 26]            16
│    └─Dropout: 2-4                      [1, 8, 26, 26]            --
│    └─Conv2d: 2-5                       [1, 16, 24, 24]           1,152
│    └─ReLU: 2-6                         [1, 16, 24, 24]           --
│    └─BatchNorm2d: 2-7                  [1, 16, 24, 24]           32
│    └─Dropout: 2-8                      [1, 16, 24, 24]           --
│    └─Conv2d: 2-9                       [1, 8, 24, 24]            128
│    └─MaxPool2d: 2-10                   [1, 8, 12, 12]            --
├─Sequential: 1-2                        [1, 16, 6, 6]             --
│    └─Conv2d: 2-11                      [1, 16, 10, 10]           1,152
│    └─ReLU: 2-12                        [1, 16, 10, 10]           --
│    └─BatchNorm2d: 2-13                 [1, 16, 10, 10]           32
│    └─Dropout: 2-14                     [1, 16, 10, 10]           --
│    └─Conv2d: 2-15                      [1, 16, 8, 8]             2,304
│    └─ReLU: 2-16                        [1, 16, 8, 8]             --
│    └─BatchNorm2d: 2-17                 [1, 16, 8, 8]             32
│    └─Dropout: 2-18                     [1, 16, 8, 8]             --
│    └─Conv2d: 2-19                      [1, 16, 6, 6]             2,304
│    └─ReLU: 2-20                        [1, 16, 6, 6]             --
│    └─BatchNorm2d: 2-21                 [1, 16, 6, 6]             32
│    └─Dropout: 2-22                     [1, 16, 6, 6]             --
├─AdaptiveAvgPool2d: 1-3                 [1, 16, 1, 1]             --
├─Sequential: 1-4                        [1, 10, 1, 1]             --
│    └─Conv2d: 2-23                      [1, 10, 1, 1]             160
==========================================================================================
Total params: 7,416
Trainable params: 7,416
Non-trainable params: 0
Total mult-adds (M): 1.13
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.32
Params size (MB): 0.03
Estimated Total Size (MB): 0.35
==========================================================================================
```

## Data Transformations

```python
transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
transforms.RandomAffine(degrees=15),
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
```


## Training Logs

```
Adjusting learning rate of group 0 to 7.5000e-02.
Epoch=0 Batch=468 loss=0.1479667 Accuracy=92.80%: 100%|██████████| 469/469 [00:10<00:00, 45.39it/s]

Test set: Average loss: 0.0003707, Accuracy: 9855/10000 (98.55%)

Adjusting learning rate of group 0 to 7.5000e-02.
Epoch=1 Batch=468 loss=0.0885876 Accuracy=97.39%: 100%|██████████| 469/469 [00:10<00:00, 44.58it/s]

Test set: Average loss: 0.0002947, Accuracy: 9886/10000 (98.86%)

Adjusting learning rate of group 0 to 4.8750e-02.
Epoch=2 Batch=468 loss=0.0137194 Accuracy=98.06%: 100%|██████████| 469/469 [00:10<00:00, 45.14it/s]

Test set: Average loss: 0.0002271, Accuracy: 9907/10000 (99.07%)

Adjusting learning rate of group 0 to 4.8750e-02.
Epoch=3 Batch=468 loss=0.0606191 Accuracy=98.16%: 100%|██████████| 469/469 [00:10<00:00, 45.67it/s]

Test set: Average loss: 0.0002112, Accuracy: 9902/10000 (99.02%)

Adjusting learning rate of group 0 to 3.1688e-02.
Epoch=4 Batch=468 loss=0.0453521 Accuracy=98.35%: 100%|██████████| 469/469 [00:10<00:00, 45.15it/s]

Test set: Average loss: 0.0001989, Accuracy: 9928/10000 (99.28%)

Adjusting learning rate of group 0 to 3.1688e-02.
Epoch=5 Batch=468 loss=0.0913020 Accuracy=98.49%: 100%|██████████| 469/469 [00:10<00:00, 45.49it/s]

Test set: Average loss: 0.0001853, Accuracy: 9924/10000 (99.24%)

Adjusting learning rate of group 0 to 2.0597e-02.
Epoch=6 Batch=468 loss=0.1289853 Accuracy=98.64%: 100%|██████████| 469/469 [00:10<00:00, 44.58it/s]

Test set: Average loss: 0.0001501, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 2.0597e-02.
Epoch=7 Batch=468 loss=0.0655444 Accuracy=98.64%: 100%|██████████| 469/469 [00:10<00:00, 44.04it/s]

Test set: Average loss: 0.0001657, Accuracy: 9929/10000 (99.29%)

Adjusting learning rate of group 0 to 1.3388e-02.
Epoch=8 Batch=468 loss=0.0361441 Accuracy=98.69%: 100%|██████████| 469/469 [00:10<00:00, 45.55it/s]

Test set: Average loss: 0.0001558, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 1.3388e-02.
Epoch=9 Batch=468 loss=0.0231169 Accuracy=98.72%: 100%|██████████| 469/469 [00:10<00:00, 44.77it/s]

Test set: Average loss: 0.0001445, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 8.7022e-03.
Epoch=10 Batch=468 loss=0.0279460 Accuracy=98.72%: 100%|██████████| 469/469 [00:10<00:00, 46.48it/s]

Test set: Average loss: 0.0001422, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 8.7022e-03.
Epoch=11 Batch=468 loss=0.1083443 Accuracy=98.79%: 100%|██████████| 469/469 [00:10<00:00, 45.03it/s]

Test set: Average loss: 0.0001459, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 5.6564e-03.
Epoch=12 Batch=468 loss=0.0243121 Accuracy=98.83%: 100%|██████████| 469/469 [00:10<00:00, 45.66it/s]

Test set: Average loss: 0.0001575, Accuracy: 9947/10000 (99.47%)

Adjusting learning rate of group 0 to 5.6564e-03.
Epoch=13 Batch=468 loss=0.0625362 Accuracy=98.87%: 100%|██████████| 469/469 [00:10<00:00, 45.20it/s]

Test set: Average loss: 0.0001386, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 3.6767e-03.
Epoch=14 Batch=468 loss=0.0151680 Accuracy=98.85%: 100%|██████████| 469/469 [00:10<00:00, 45.68it/s]

Test set: Average loss: 0.0001644, Accuracy: 9934/10000 (99.34%)

Adjusting learning rate of group 0 to 3.6767e-03.
```


# [MnistNet_3](MNIST_S7_T3.ipynb)

## Target
- Reduce parameters to less than 7K
- Batch normalization and regularization
- Play around learning rate and scheduler configuration & dropout rate
- Achieve 99.4% test accuracy at least once before 10th epoch

## Result
- Parameters: 6,544
- Best Training Accuracy: 98.92%
- Best Test Accuracy: 99.45%

## Analysis
- Consistent test accuracy i.e., more than 99.4% from 10th epoch onwards
- dropout rate: 0.01, seems to perform better than 0.05
- Train-test gap can be seen as train accuracy doesn't reach 99%


## Network Architecture 

| INPUT    | KERNEL     | OUTPUT |
|----------|------------|--------|
| 28x28x1  | 3x3x1x8    | 26     |
| 26x26x8  | 3x3x8x16   | 24     |
| 24x24x16 | 1x1x16x8   | 24     |
| Maxpool()|            | 12     |
| 12x12x8  | 3x3x8x12   | 10     |
| 10x10x12 | 3x3x12x16  | 8      |
| 8x8x16   | 3x3x16x16  | 6      |
| AdaptiveAvgPool()|    | 1      | 
| 1x1x16   | 1x1x16x10  | 1      |
| Flatten()             |        | 


## RF Calculation

**Formula of RF**: \(r_{out}\) = \(r_{in}\) + (\(k\) - 1) * \(j_{in}\)

**Formula of Jump**: \(j_{out}\) = \(j_{in}\) * \(s\)


| \(n_{in}\) | \(k\) | \(s\) | \(p\) | \(n_{out}\) | \(r_{in}\) | \(j_{in}\) | \(r_{out}\) | \(j_{out}\) |
|------------|-------|-------|-------|-------------|------------|------------|-------------|-------------|
| 28         | 3     | 1     | 0     | 26          | 1          | 1          | 3           | 1           |
| 26         | 3     | 1     | 0     | 24          | 3          | 1          | 5           | 1           |
| 24         | 1     | 1     | 0     | 24          | 5          | 1          | 5           | 1           |
| 24         | 2     | 2     | 0     | 12          | 5          | 1          | 6           | 2           |
| 12         | 3     | 1     | 0     | 10          | 6          | 2          | 10          | 2           |
| 10         | 3     | 1     | 0     | 8           | 10         | 2          | 14          | 2           |
| 8          | 3     | 1     | 0     | 6           | 14         | 2          | 18          | 2           |


## Model Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MnistNet_3                               [1, 10]                   --
├─Sequential: 1-1                        [1, 8, 12, 12]            --
│    └─Conv2d: 2-1                       [1, 8, 26, 26]            72
│    └─BatchNorm2d: 2-2                  [1, 8, 26, 26]            16
│    └─ReLU: 2-3                         [1, 8, 26, 26]            --
│    └─Dropout: 2-4                      [1, 8, 26, 26]            --
│    └─Conv2d: 2-5                       [1, 16, 24, 24]           1,152
│    └─BatchNorm2d: 2-6                  [1, 16, 24, 24]           32
│    └─ReLU: 2-7                         [1, 16, 24, 24]           --
│    └─Dropout: 2-8                      [1, 16, 24, 24]           --
│    └─Conv2d: 2-9                       [1, 8, 24, 24]            128
│    └─MaxPool2d: 2-10                   [1, 8, 12, 12]            --
├─Sequential: 1-2                        [1, 16, 6, 6]             --
│    └─Conv2d: 2-11                      [1, 12, 10, 10]           864
│    └─ReLU: 2-12                        [1, 12, 10, 10]           --
│    └─BatchNorm2d: 2-13                 [1, 12, 10, 10]           24
│    └─Dropout: 2-14                     [1, 12, 10, 10]           --
│    └─Conv2d: 2-15                      [1, 16, 8, 8]             1,728
│    └─ReLU: 2-16                        [1, 16, 8, 8]             --
│    └─BatchNorm2d: 2-17                 [1, 16, 8, 8]             32
│    └─Dropout: 2-18                     [1, 16, 8, 8]             --
│    └─Conv2d: 2-19                      [1, 16, 6, 6]             2,304
│    └─ReLU: 2-20                        [1, 16, 6, 6]             --
│    └─BatchNorm2d: 2-21                 [1, 16, 6, 6]             32
│    └─Dropout: 2-22                     [1, 16, 6, 6]             --
├─AdaptiveAvgPool2d: 1-3                 [1, 16, 1, 1]             --
├─Sequential: 1-4                        [1, 10, 1, 1]             --
│    └─Conv2d: 2-23                      [1, 10, 1, 1]             160
==========================================================================================
Total params: 6,544
Trainable params: 6,544
Non-trainable params: 0
Total mult-adds (M): 1.07
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.32
Params size (MB): 0.03
Estimated Total Size (MB): 0.35
==========================================================================================
```

## Data Transformations

```python
transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
transforms.RandomAffine(degrees=15)
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
```


## Training Logs

```
Adjusting learning rate of group 0 to 5.0000e-02.
Epoch=0 Batch=468 loss=0.0506339 Accuracy=90.56%: 100%|██████████| 469/469 [00:10<00:00, 46.40it/s]

Test set: Average loss: 0.0007933, Accuracy: 9711/10000 (97.11%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch=1 Batch=468 loss=0.0882496 Accuracy=97.37%: 100%|██████████| 469/469 [00:10<00:00, 46.03it/s]

Test set: Average loss: 0.0003369, Accuracy: 9859/10000 (98.59%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch=2 Batch=468 loss=0.1382393 Accuracy=97.76%: 100%|██████████| 469/469 [00:10<00:00, 46.23it/s]

Test set: Average loss: 0.0003078, Accuracy: 9869/10000 (98.69%)

Adjusting learning rate of group 0 to 5.0000e-02.
Epoch=3 Batch=468 loss=0.1314877 Accuracy=98.00%: 100%|██████████| 469/469 [00:10<00:00, 45.36it/s]

Test set: Average loss: 0.0002391, Accuracy: 9900/10000 (99.00%)

Adjusting learning rate of group 0 to 1.7500e-02.
Epoch=4 Batch=468 loss=0.0214672 Accuracy=98.47%: 100%|██████████| 469/469 [00:10<00:00, 45.88it/s]

Test set: Average loss: 0.0001982, Accuracy: 9924/10000 (99.24%)

Adjusting learning rate of group 0 to 1.7500e-02.
Epoch=5 Batch=468 loss=0.0650054 Accuracy=98.65%: 100%|██████████| 469/469 [00:10<00:00, 45.56it/s]

Test set: Average loss: 0.0001928, Accuracy: 9917/10000 (99.17%)

Adjusting learning rate of group 0 to 1.7500e-02.
Epoch=6 Batch=468 loss=0.0137649 Accuracy=98.62%: 100%|██████████| 469/469 [00:10<00:00, 45.43it/s]

Test set: Average loss: 0.0001876, Accuracy: 9924/10000 (99.24%)

Adjusting learning rate of group 0 to 1.7500e-02.
Epoch=7 Batch=468 loss=0.0361735 Accuracy=98.62%: 100%|██████████| 469/469 [00:10<00:00, 45.43it/s]

Test set: Average loss: 0.0001644, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 6.1250e-03.
Epoch=8 Batch=468 loss=0.0236110 Accuracy=98.69%: 100%|██████████| 469/469 [00:10<00:00, 46.04it/s]

Test set: Average loss: 0.0001676, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 6.1250e-03.
Epoch=9 Batch=468 loss=0.0207121 Accuracy=98.78%: 100%|██████████| 469/469 [00:10<00:00, 45.71it/s]

Test set: Average loss: 0.0001554, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 6.1250e-03.
Epoch=10 Batch=468 loss=0.0341703 Accuracy=98.80%: 100%|██████████| 469/469 [00:10<00:00, 44.86it/s]

Test set: Average loss: 0.0001628, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 6.1250e-03.
Epoch=11 Batch=468 loss=0.0304645 Accuracy=98.78%: 100%|██████████| 469/469 [00:10<00:00, 45.61it/s]

Test set: Average loss: 0.0001602, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 2.1437e-03.
Epoch=12 Batch=468 loss=0.0260025 Accuracy=98.92%: 100%|██████████| 469/469 [00:10<00:00, 45.42it/s]

Test set: Average loss: 0.0001555, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 2.1437e-03.
Epoch=13 Batch=468 loss=0.0471064 Accuracy=98.91%: 100%|██████████| 469/469 [00:10<00:00, 45.59it/s]

Test set: Average loss: 0.0001560, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 2.1437e-03.
Epoch=14 Batch=468 loss=0.0130037 Accuracy=98.88%: 100%|██████████| 469/469 [00:10<00:00, 45.47it/s]

Test set: Average loss: 0.0001537, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 2.1437e-03.
```

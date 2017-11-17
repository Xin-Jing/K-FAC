# KFAC
Implementation of KFAC in Python. Please see paper https://arxiv.org/pdf/1503.05671.pdf

So far, this KFAC works for a feedfowrd neural network doing classification. Only block-diagonal approximation of the Fisher matrix is implemented. Will improve it soon.

The file "Experiments" is used as a comparison to the performance of KFAC. It performs exactly the same classification task with an adam optimizer. It has the same backpropagation code as that for KFAC. So it differs from KFAC in what happens after backpropagation.

# K-FAC
Implementation of K-FAC in Python. Please see paper https://arxiv.org/pdf/1503.05671.pdf

So far, this K-FAC works for a feedfowrd neural network doing classification. Only block-diagonal approximation of the Fisher matrix is implemented. Will improve it soon.

The file "Experiments" is used as a comparison to the performance of K-FAC. It performs exactly the same classification task with an adam optimizer. It has the same backpropagation code as that for K-FAC. So it differs from K-FAC in what happens after backpropagation.

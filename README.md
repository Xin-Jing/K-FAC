# K-FAC

The Implementation of K-FAC. Please see the excellent paper for details: https://arxiv.org/pdf/1503.05671.pdf

K-FAC is short for "Kronecker-factored Approximate Curvature". It optimizes neural networks using block-diagonal approximation to the Fisher information matrix required in the Natural Gradient algorithm. It converges with far fewer iterations than other algorithms such as SGD and adam optimizer.

So far, only feedforward neural networks are supported. If you are training a MLP, please try it!

This implementation is simple to use. Users don't need to construct the neural networks on their own. By simply specifying the activation type and size of each layer, the optimizer will construct the neural network inside (see [Example](Example.py)). 

The file [optimize_with_adam](optimize_with_adam.py) is used as a comparison to the performance of K-FAC. It performs exactly the same classification task with an adam optimizer. It has the same backpropagation code as that of K-FAC. So it differs from K-FAC in what happens after backpropagation.
The

I'm continuing improving the implementation. If you have any suggestion, please feel free to contact me at xin.jing@mail.utoronto.ca

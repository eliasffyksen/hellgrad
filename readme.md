# *HellGrad*

> Neural Networks and Automatic Differentiation in Haskell...
>
> ...how bad can it be?

An auto grad and neural network framework implemented in Haskell.
Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd) and [google/JAX](https://github.com/google/jax)

* `Grad.hs` - *82 lines*
  * Scalar auto grad engine
* `Vec.hs` - *23 lines*
  * Vectorisation of `Grad`
* `NN.hs` - *49 lines*
  * Neural Network Framework

Example:
```hs
import qualified NN
import qualified Grad

(params, forward) = NN.sequential [
  NN.linear 2 16,
  NN.bias 16,
  NN.relu,
  NN.linear 16 2 ]

x = get_input
y = get_targets

y_pred = forward (Grad.grad params) x
loss = NN.mse y y_pred
param_grads = Grad.backward loss

new_params = Vec.sub params params_grad
```

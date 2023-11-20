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

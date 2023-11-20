module NN where

import qualified Grad
import qualified Vec

type Params = [Grad.Value]
type ForwardFn a = Params -> a -> a
type Module a = (Params, ForwardFn a)

vectorise :: Module a -> Module [a]
vectorise (params, org_forward) =
  let forward params = Vec.vectorise (org_forward params)
  in (params, forward)

relu :: Module [Grad.Value]
relu = ([], \_ -> Vec.relu)

bias :: Int -> Module [Grad.Value]
bias size =
  let params = take size (repeat $ Grad.value 0.0)
  in let forward params = Vec.add $ take size params
  in (params, forward)

_linearForward :: Int -> Int -> ForwardFn [Grad.Value]
_linearForward in_size 0 params x = []
_linearForward in_size out_size params x = Vec.dot x (take in_size params) :
  _linearForward in_size (out_size - 1) (drop in_size params) x

linear :: Int -> Int -> Module [Grad.Value]
linear in_size out_size =
  let params = take (in_size * out_size) $ repeat $ Grad.value 0.0
  in let forward = _linearForward in_size out_size
  in (params, forward)

chainForward :: Int -> (ForwardFn a) -> (ForwardFn a) -> ForwardFn a
chainForward a_param_size fn_a fn_b params x =
  let z = fn_a (take a_param_size params) x
  in fn_b (drop a_param_size params) z

sequential :: [Module a] -> Module a
sequential [] = ([], \params -> \x -> x)
sequential (mod:mods) =
  let (params_rest, next_forward) = sequential mods
  in let (params, this_forward) = mod
  in let param_size = length params
  in let forward = chainForward param_size this_forward next_forward
  in (params ++ params_rest, forward)

mse y y_pred = Vec.sum $ Vec.pow 2.0 (Vec.sub y y_pred)

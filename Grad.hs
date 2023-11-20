module Grad where

import Data.List (sortBy)
import Data.Ord (comparing)

type GradPool = [(Int, Double)]

type BackwardFn = GradPool -> Double -> GradPool

data Value = Value {
  _value :: Double,
  _backward :: BackwardFn }

instance Show (Value) where
  show v = show $ _value v

value :: Double -> Value
value n = Value {
  _value = n,
  _backward = \pool -> \grad -> pool }

gradBackward :: Int -> (BackwardFn) -> BackwardFn
gradBackward i org_backward pool grad =
  org_backward ((i, grad) : pool) grad

_grad :: [Value] -> Int -> [Value]
_grad [] _ = []
_grad (x:xs) i = Value {
  _value = _value x,
  _backward = gradBackward i $ _backward x
} : (_grad xs (i + 1))

grad ::  [Value] -> [Value]
grad v = _grad v 0

sortGradPool :: GradPool -> GradPool
compareGradPools (a, _) (b, _) = compare a b
sortGradPool = sortBy compareGradPools

poolToValues :: GradPool -> Int -> Double -> [Value]
poolToValues [] _ d = [value d]
poolToValues ((i, d) : pool) j d_old = if j < i
  then (value d_old) : poolToValues ((i, d) : pool) (j + 1) 0.0
  else poolToValues pool j (d + d_old)

backward :: Value -> [Value]
backward v = poolToValues (sortGradPool $ _backward v [] 1.0) 0 0.0

backwardCompose :: [BackwardFn] -> BackwardFn
backwardCompose [] pool _ = pool
backwardCompose (fn:fns) pool grad =
  let back_rest = backwardCompose fns
  in fn (back_rest pool grad) grad

add :: Value -> Value -> Value
add a b = Value {
  _value = (_value a) + (_value b),
  _backward = backwardCompose [ _backward a, _backward b ]
}

mul :: Value -> Value -> Value
mul a b = Value {
  _value = (_value a) * (_value b),
  _backward = backwardCompose [
    \pool -> \grad -> _backward a pool $ grad * _value b,
    \pool -> \grad -> _backward b pool $ grad * _value a
  ]
}

relu :: Value -> Value
relu a = if (_value a) < 0.0
  then value $ _value a
  else a

sub a b = add a $ mul b $ value $ -1.0

pow :: Double -> Value -> Value
pow exponent x = Value {
  _value = (_value x) ** exponent,
  _backward = \pool -> \grad ->
    _backward x pool (exponent * (_value x) ** (exponent - 1) )
}


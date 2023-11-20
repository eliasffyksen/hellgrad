module Vec where

import qualified Grad

opp :: (a -> a -> a) -> [a] -> [a] -> [a]
opp fn (a:[]) (b:[]) = [fn a b]
opp fn (a:as) (b:bs) = (fn a b) : opp fn as bs

reduce :: (a -> a -> a) -> a -> [a] -> a
reduce fn acc [] = acc
reduce fn acc (x:xs) = reduce fn (fn acc x) xs

vectorise :: (a -> b) -> [a] -> [b]
vectorise fn [] = []
vectorise fn (x:xs) = (fn x) : vectorise fn xs

add = opp Grad.add
sub = opp Grad.sub
mul = opp Grad.mul
pow d = vectorise (Grad.pow d)
relu = vectorise Grad.relu
sum = reduce Grad.add (Grad.value 0.0)
dot a b = Vec.sum (mul a b)

# Linnaeus5 Transformer Image Classifier

The Linneaus 5 Dataset can be found [here](http://chaladze.com/l5/). I've used the 64x64 resolution version.

This repo contains a basic image classifier with a transformer for encoding. The image is broken up into patches of 16x16 - which neatly divides the 64x64 images - and then feeds the flattened sequence into a transformer encoder layer. 
The decoder is a feed forward network that outputs the class logits. 


import pySTL
from numpy import array

#Load a model from a file.
model = pySTL.STLmodel('model_dla_bartka_bez_skali.stl')

scale = 0.00035
model.scale(scale)

model.write_text_stl('Bartek_skala_2857.stl')


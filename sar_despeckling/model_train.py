from model import build_model
from data_generator import generator

model = build_model()
gen = generator()

model.fit(gen, steps_per_epoch=840, epochs=3)
model.save("C:/Users/tim.iles/noise_model_noise2.h5")
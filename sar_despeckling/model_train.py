from model import dilation_net
from data_generator import generator

model = dilation_net()
gen = generator()
model.fit(gen, steps_per_epoch=3000, epochs=3)
model.save("C:/Users/tim.iles/noise_model_noise_synthetic_sv_loss.h5")
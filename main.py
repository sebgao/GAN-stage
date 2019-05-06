import models
from models.utils import instantiate
model_dict = dict(
    type='DCGAN',
    noise = dict(
        type='GaussianNoisyGenerator',
        bandwidth=3.0,
        batch=2,
    )
)

dataset = dict(
    type='StandardImageFolder',
    root='faceA/'
)

model = instantiate(model_dict)

model.config(dataset, batch_size=4)
model.start_train()
print(model)
#print(model().shape)
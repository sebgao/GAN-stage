import models
from models.utils import instantiate
model = dict(
    type='DCGAN',
    noise = dict(
        type='UniformNoisyGenerator',
        batch=2,
    )
)
print(instantiate(model))
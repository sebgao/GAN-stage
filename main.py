import models

#noise = registry.GENERATOR['UniformNoisyGenerator']()
#generator = registry.GENERATOR['DCGenerator']()
#print(generator(noise(None)))
noise = models.LIBRARY.UniformNoisyGenerator()
generator = models.LIBRARY.DCGenerator(start_size=(4, 4), levels=4)
discriminator = models.LIBRARY.DCDiscriminator()

print(discriminator(generator(noise())).shape)

print(discriminator)

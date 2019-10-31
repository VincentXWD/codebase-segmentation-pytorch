from models import Unet


def get_model():
  return Unet(
      encoder_name='dilated_resnet101',
      encoder_weights='imagenet',
      decoder_use_batchnorm=True,
      classes=19)

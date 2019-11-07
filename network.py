from models import Unet
from models import PSPNet

# Unet. Please note that only support edge size is 4 times power of 2.
# def get_model(criterion=None, auxiliary_loss=False, auxloss_weight=0):
#   return Unet(
#       encoder_name='dilated_resnet101',
#       encoder_weights='imagenet',
#       decoder_use_batchnorm=True,
#       classes=19,
#       criterion=criterion)

# PSPNet
def get_model(criterion=None, auxiliary_loss=False, auxloss_weight=0):
  return PSPNet(
      encoder_name='dilated_resnet101',
      encoder_weights='imagenet',
      classes=19,
      auxiliary_loss=auxiliary_loss,
      auxloss_weight=auxloss_weight,
      criterion=criterion)

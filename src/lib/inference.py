"""
Methods for performing forward pass and inference through the model

@author: Angel Villar-Corrales
"""

import lib.transforms as custom_transforms
import CONSTANTS


def forward_pass(model, img, model_name, device=None, flip=False):
    """
    Computing a forward pass throught the model, given the name of the architecture
    """

    if(model_name == "HRNet"):
        # forward pass for input image
        output = model(img)
        # forward pass for flipped image
        if(flip is True):
            flipped_img = img.flip(3)
            output_flipped = model(flipped_img)
            output_flipped = custom_transforms.flip_back(output_flipped, CONSTANTS.FLIP_PAIRS)
            output_flipped = output_flipped.to(device)
            output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5  # averaging both predicitions
        return output

    else:
        raise NotImplementedError("Wrong model name. Only ['HRNet'] supported")

    return

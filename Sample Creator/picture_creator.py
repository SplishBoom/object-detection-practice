"""
This program creates pictures by using the images under Object_Detection/ directory.
"""

from calendar import c
import os
import numpy as np
from PIL import Image
import PIL

objectCount = 5
initialDirectory = "Object Photographs/"

objectImages = [Image.open(initialDirectory + objName) for objName in np.random.choice(os.listdir(initialDirectory), objectCount, replace=False)]

minImageShape = sorted([np.sum(i.size), i.size] for i in objectImages)[0][1]

horizontalCombination = np.hstack((np.asarray(i.resize(minImageShape)) for i in objectImages))
horizontalCombination = Image.fromarray(horizontalCombination)
horizontalCombination.save("Picture.jpg")
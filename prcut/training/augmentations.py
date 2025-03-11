import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import Transform
import torch
import random
import math


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    """

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=[0.4914, 0.4822, 0.4465],
        n_attempts=100,
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.n_attempts = n_attempts

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(self.n_attempts):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                    img[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                    img[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                else:
                    img[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                return img

        return img


class ReshapeAsImage(Transform):

    def __init__(self, shape):
        self.shape = shape
        self.need_reshape = False
        if len(shape) == 2:
            self.need_reshape = True

    def __call__(self, images):
        if self.need_reshape:
            images = images.view(images.shape[0], -1, *self.shape)
        return images


class Reshape(Transform):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, images):
        return images.view(images.shape[0], *self.shape)


class SimCLRAugmentation(Transform):
    def __init__(self, shape, jitter_rate=1.0, training=True):
        super().__init__()
        if training:
            self.data_transforms = transforms.Compose(
                [
                    # ReshapeAsImage(shape),
                    transforms.RandomResizedCrop(size=shape),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                0.8 * jitter_rate,
                                0.8 * jitter_rate,
                                0.8 * jitter_rate,
                                0.2 * jitter_rate,
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(
                        kernel_size=[(int(s * 0.1) // 2) * 2 + 1 for s in shape]
                    ),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.data_transforms = transforms.ToDtype(torch.float32, scale=True)

    def __call__(self, inp):
        return self.data_transforms(inp)

    def __repr__(self):
        return super().__repr__() + ": SimCLR Transform"

        # transforms.Normalize((0.1307,), (0.3081,)),


#         transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3, mean=[0.4914]),

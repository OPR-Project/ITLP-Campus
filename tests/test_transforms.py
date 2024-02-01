import numpy as np

from itlp_campus.transforms import DefaultImageTransform


def test_default_image_transform_train():
    transform = DefaultImageTransform(train=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    transformed_image = transform(image)
    assert transformed_image.shape == (3, 100, 100)


def test_default_image_transform_not_train():
    transform = DefaultImageTransform(train=False)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    transformed_image = transform(image)
    assert transformed_image.shape == (3, 100, 100)


def test_default_image_transform_resize():
    transform = DefaultImageTransform(resize=(50, 50))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    transformed_image = transform(image)
    assert transformed_image.shape == (3, 50, 50)

from PIL import Image

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128


def numpy_array_to_black_and_white_images(images):
    return [Image.fromarray(image).convert('L') for image in images]


def channels_to_rgb(channels, file_name):
    assert len(channels) == 3, 'Three channels are required to convert to rgb'

    black_and_white_images = numpy_array_to_black_and_white_images(channels)
    red, green, blue = black_and_white_images

    result = Image.merge('RGB', (red, green, blue))
    result.save(file_name)


def stack_frames(images, file_name):
    black_and_white_images = numpy_array_to_black_and_white_images(images)

    stacked_image = Image.new('RGB', (IMAGE_WIDTH, 3 * IMAGE_HEIGHT))

    y_offset = 0
    for channel in [black_and_white_images]:
        stacked_image.paste(channel, (0, y_offset))
        y_offset += IMAGE_HEIGHT

    stacked_image.save(file_name)

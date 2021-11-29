import numpy as np
from PIL import Image

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128


def greyscale_to_rgb(video_number, frame_number):
    image_paths = {
        'ambient': 'data/all_videos/images/Video00000_ambient_frame_000000.jpg',
        'intensity': 'data/all_videos/images/Video00000_intensity_frame_000000.jpg',
        'range': 'data/all_videos/images/Video00000_range_frame_000000.jpg',
    }
    ambient = Image.open(image_paths['ambient']).convert('L')
    intensity = Image.open(image_paths['intensity']).convert('L')
    range = Image.open(image_paths['range']).convert('L')

    result = Image.merge('RGB', (ambient, intensity, range))
    result.save('rgb.jpg')


def stack_frames(video_number, frame_number):
    image_paths = {
        'ambient': 'data/all_videos/images/Video00000_ambient_frame_000000.jpg',
        'intensity': 'data/all_videos/images/Video00000_intensity_frame_000000.jpg',
        'range': 'data/all_videos/images/Video00000_range_frame_000000.jpg',
    }
    ambient = Image.open(image_paths['ambient']).convert('L')
    intensity = Image.open(image_paths['intensity']).convert('L')
    range = Image.open(image_paths['range']).convert('L')

    stacked_image = Image.new('RGB', (IMAGE_WIDTH, 3 * IMAGE_HEIGHT))

    y_offset = 0
    for image in [ambient, intensity, range]:
        stacked_image.paste(image, (0, y_offset))
        y_offset += IMAGE_HEIGHT

    stacked_image.save('stacked.jpg')


if __name__ == '__main__':
    greyscale_to_rgb('Video00000', 'frame_000000')

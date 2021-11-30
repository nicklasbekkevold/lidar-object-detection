import os
import sys

import cv2

from image_transformations import channels_to_rgb


class DatasetBuilder:

    def __init__(self):
        self.video_numbers_sorted = []

    def convert_to_frames(self, video_number, video_file):
        parsed_file_name = video_file.split('/')[-1].split('.')[0]
        video_capture = cv2.VideoCapture(video_file)

        frame_number = 0

        success, image = video_capture.read()
        while success:
            # These needs to match
            image_file_name = f'data/all_videos/images/{parsed_file_name}_frame_{frame_number:06}.jpg'
            label_file_name = f'data/all_videos/labels/{parsed_file_name}_frame_{frame_number:06}.txt'

            cv2.imwrite(image_file_name, image)  # save frame as JPEG file
            os.popen(f'cp {self.label_paths[video_number][frame_number]} {label_file_name}')  # copy corresponding label

            success, image = video_capture.read()
            frame_number += 1

        print('Finished converting:', parsed_file_name)

    def convert_videos_to_frames(self):
        for video_number, video_files in self.video_paths.items():
            for video_file in video_files:
                self.convert_to_frames(video_number, video_file)

    def convert_to_frames_and_combine(self, video_number, video_files):
        parsed_file_name = video_files[0].split('/')[-1].split('_')[0]
        video_captures = list(map(cv2.VideoCapture, sorted(video_files)))

        frame_number = 0

        successes, channels = list(zip(*map(cv2.VideoCapture.read, video_captures)))
        while all(successes):
            # These needs to match
            image_file_name = f'data/all_videos_merged/images/{parsed_file_name}_frame_{frame_number:06}.jpg'
            label_file_name = f'data/all_videos_merged/labels/{parsed_file_name}_frame_{frame_number:06}.txt'

            channels_to_rgb(channels, image_file_name)  # save frame as JPEG file
            os.popen(f'cp {self.label_paths[video_number][frame_number]} {label_file_name}')  # copy corresponding label

            successes, channels = list(zip(*map(cv2.VideoCapture.read, video_captures)))
            frame_number += 1

        print('Finished converting and merging:', parsed_file_name)

    def convert_videos_to_frames_and_combine(self):
        for video_number, video_files in self.video_paths.items():
            self.convert_to_frames_and_combine(video_number, video_files)

    def parse_video_label_directories(self):
        # Fetch labeled video numbers
        _, dirs, _ = next(os.walk('./data', topdown=True))
        self.video_label_dir_names = {}
        for name in dirs:
            if name not in ['all_videos', 'all_videos_merged']:
                self.video_label_dir_names[int(name.split('_')[0])] = name

        self.video_numbers_sorted = sorted(self.video_label_dir_names.keys())

    def get_num_of_label_files(self):
        _, directories, _ = next(os.walk('./data', topdown=True))
        self.num_of_label_files = {video_number: 0 for video_number in self.video_numbers_sorted}
        for directory in directories:
            if directory not in ['all_videos', 'all_videos_merged']:
                _, _, files = next(os.walk(f'./data/{directory}/obj_train_data', topdown=True))
                self.num_of_label_files[int(directory.split('_')[0])] = len(files)

    def generate_label_paths(self):
        self.label_paths = {video_number: [f'./data/{self.video_label_dir_names[video_number]}/obj_train_data/frame_{count:06}.txt' for count in range(
            self.num_of_label_files[video_number])] for video_number in self.video_numbers_sorted}

    def get_video_paths(self):
        # Fetch video files
        path, _, files = next(os.walk('./videos', topdown=True))
        self.video_paths = {video_number: []
                            for video_number in self.video_numbers_sorted}
        for name in files:
            video_number = int(name.split('_')[0].split('Video')[1])
            if video_number in self.video_numbers_sorted:
                self.video_paths[video_number].append(os.path.join(path, name))

    def make_dirs_if_not_exists(self, folder_name):
        try:
            os.makedirs(f'data/{folder_name}/images')
            os.mkdir(f'data/{folder_name}/labels')
        except OSError:
            pass

    def build(self, combine_channels):
        folder_name = 'all_videos_merged' if combine_channels else 'all_videos'

        self.parse_video_label_directories()
        self.get_num_of_label_files()
        self.generate_label_paths()
        self.get_video_paths()
        self.make_dirs_if_not_exists(folder_name)

        if combine_channels:
            self.convert_videos_to_frames_and_combine()
        else:
            self.convert_videos_to_frames()


if __name__ == '__main__':
    combine_channels = len(sys.argv) > 1 and sys.argv[1] == '--merge'

    dataset_builder = DatasetBuilder()
    dataset_builder.build(combine_channels=combine_channels)

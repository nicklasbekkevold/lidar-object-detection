import cv2
import os


class DatasetBuilder:

    def __init__(self):
        self.video_numbers_sorted = []

    def convert(self, video_number, video_file):
        parsed_file_name = video_file.split('/')[-1].split('.')[0]
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        while success:
            framecount = "{number:06}".format(number=count)
            cv2.imwrite(f"data/all_videos/images/{parsed_file_name}_frame_{framecount}.jpg",
                        image)     # save frame as JPEG file
            os.popen(
                f'cp {self.label_paths[video_number][count]} data/all_videos/labels/{parsed_file_name}_frame_{framecount}.txt')
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

    def parse_video_label_directories(self):
        # Fetch labeled video numbers
        _, dirs, _ = next(os.walk("./data", topdown=True))
        self.video_label_dir_names = {}
        for name in dirs:
            if name == "all_videos":
                continue
            self.video_label_dir_names[int(name.split("_")[0])] = name
        self.video_numbers_sorted = sorted(self.video_label_dir_names.keys())

    def get_num_of_label_files(self):
        _, dirs, _ = next(os.walk("./data", topdown=True))
        self.num_of_label_files = {
            video_number: 0 for video_number in self.video_numbers_sorted}
        for dir in dirs:
            if dir == "all_videos":
                continue
            _, _, files = next(
                os.walk(f"./data/{dir}/obj_train_data", topdown=True))
            self.num_of_label_files[int(dir.split("_")[0])] = len(files)

    def generate_label_paths(self):
        self.label_paths = {video_number: [f"./data/{self.video_label_dir_names[video_number]}/obj_train_data/frame_{'{number:06}'.format(number=count)}.txt" for count in range(
            self.num_of_label_files[video_number])] for video_number in self.video_numbers_sorted}

    def get_video_paths(self):
        # Fetch video files
        path, _, files = next(os.walk("./videos", topdown=True))
        self.video_paths = {video_number: []
                            for video_number in self.video_numbers_sorted}
        for name in files:
            video_number = int(name.split("_")[0].split("Video")[1])
            if video_number in self.video_numbers_sorted:
                self.video_paths[video_number].append(os.path.join(path, name))

    def make_dirs_if_not_exists(self):
        try:
            os.makedirs(f"data/all_videos/images")
            os.mkdir(f"data/all_videos/labels")
        except OSError as error:
            pass

    def build(self):
        self.parse_video_label_directories()
        self.get_num_of_label_files()
        self.generate_label_paths()
        self.get_video_paths()
        self.make_dirs_if_not_exists()

        for video_number, video_files in self.video_paths.items():
            for video_file in video_files:
                self.convert(video_number, video_file)


if __name__ == "__main__":
    dataset_builder = DatasetBuilder()
    dataset_builder.build()

# LiDAR object detection project

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./NTNU_black_and_white.png">
  <img align="center" alt="NTNU logo" src="./NTNU.png">
</picture>

## How to run

This project is run using personal notebooks contained in the `notebooks`-folder.

The final delivery notebook is contained in [final.ipynb](notebooks/final.ipynb)

YOLOv5 has been the main architecture used up until this point.

### Generating training data

In order to train on the LiDAR videos, they first need to be converted to images (frame by frame).  
In the `src`-directory there is a script for doing exactly this.

```python
python src/dataset_builder.py [--merge] [--patches]
```

The `--merge` flag is optional. When used it merges the three video channels (ambient, intensity and range) into RGB-images.
The `--patches` flag is also optional. When used it splits all videos into 8x 128x128 images

The script can also be run via the notebook.

### Requirements

Please refer to the [YOLOv5 documentation](https://github.com/ultralytics/yolov5) for installation and use.

Python scripts has been developed using version Python 3.8, but is expected to work with other versions as well

If you are using pip, run the following at the command-line to install the project dependencies:

```shell
pip install -r requirements.txt
```

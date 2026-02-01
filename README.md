# Computer Vision Project

Click [HERE](docs/Report_Arnautovic_Kumar_Schiroli.pdf) to see the report.

<p align="center">
  <a href="https://www.youtube.com/watch?v=bemUXTbVydY">
    <img src="https://img.youtube.com/vi/bemUXTbVydY/0.jpg" width="700">
  </a>
</p>

*Click the image above to watch the full project walkthrough.*

This project takes in input a video and its output is another video with painting and people detected, paintings retrieved, rectified and people localized.


## Installation

For this project is necessary to install PyTorch and torchivision:
```bash
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
```

To install other libraries type this command:
```bash
pip install -r requirements.txt
```


## Downloading auxiliary data

Heavy files of the projects are inside this link:

https://www.dropbox.com/sh/sgli626r815ilkl/AABKlDJ5_3D_S4Hk8xm7anYUa?dl=0

or

https://drive.google.com/drive/folders/1Adhc7T-M_1epWVoltLGPYn5A3emMQMIa?usp=sharing

(to acces the second one you need unimore account)

Files are collocated in the deirectory where they should be. 
For example: 
* features.json should be located in the PaintingDetection directory
* the folder paintings_db should be inside PeopleRetrieval.
* ...


## Usage

```bash
python main.py --video --input --output --showrect
```
Meaning of each parameter:

* --video: If false, it takes an image in input
* --input: Input video path
* --output: Output video path without .avi extension
* --showrect: If true, it shows the rectification in output



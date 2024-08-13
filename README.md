# Traffic_Density
This project is a vehicle detector built using the YOLOv5 model. The following instructions will guide you on how to set up and train the model.

## Installation

First, clone the YOLOv5 repository and install the required packages:

install 
```sh
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt
pip install -q roboflow
```

train command
``` sh
python {yolo folder}/train.py --img 640 --batch 16 --epochs 150 --data {data folder}/data.yaml --weights yolov5m.pt --cache
```
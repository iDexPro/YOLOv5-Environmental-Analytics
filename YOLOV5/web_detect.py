import gradio as gr
from PIL import Image

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *

from basic_detect import detect_batch

def predict(img):
    out ="websource_in"
    # save img to file
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)
    im = Image.fromarray(img)
    im.save(out+"/tmp.jpeg")

    out2 ="websource_out"

    if not os.path.exists(out2):
        os.makedirs(out2)

    weights = "yolov5c.pt"
    source= out
    output = out2
    device = ""
    view_img = False
    save_txt = True
    imgsz = 200
    conf_thres = 0.4 
    iou_thres = 0.5
    
    im0 = detect_batch(weights,source,output,device, view_img, save_txt, imgsz, conf_thres, iou_thres)

    return im0

# webcam = gr.inputs.Webcam(shape=(400, 400))
# gr.Interface(fn=predict, inputs=webcam, outputs="image",capture_session=True,).launch(inbrowser=True)

gr.Interface(predict, gr.inputs.Image(shape=(200, 200)), "image").launch()


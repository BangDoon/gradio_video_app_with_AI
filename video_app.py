import gradio as gr
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import cv2
from torchvision import transforms
import os
from tqdm import tqdm

model = PRN(6, 1)
model = model.cuda()
model.load_state_dict(torch.load('./net_latest.pth'))
iscuda = torch.cuda.is_available()

data_path = './datasets/'
video_list= [data_path+file_name for file_name in os.listdir(data_path)]

def deNosing(noise_img):
    b, g, r = cv2.split(noise_img)
    y = cv2.merge([r, g, b])
    y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = Variable(torch.Tensor(y))

    if iscuda:
        y = y.cuda()

    with torch.no_grad(): #
        if iscuda:
            torch.cuda.synchronize()

        out, _ = model(y)
        out = torch.clamp(out, 0., 1.)

        if iscuda:
            torch.cuda.synchronize()


    if iscuda:
        save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
    else:
        save_out = np.uint8(255 * out.data.numpy().squeeze())

    save_out = save_out.transpose(1, 2, 0)
    
    return save_out

def read_video(video_path):
    paused = False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video")
        return 0
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter('./output_video/cache.mp4',fourcc ,frame_rate,(frame_width,frame_height))

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            break

        denosing_frame = deNosing(frame)
        denosing_frame = cv2.cvtColor(denosing_frame, cv2.COLOR_RGB2BGR)

        denosing_frame = cv2.resize(frame, dsize=(frame_width, frame_height), fx=0, fy=0) 

        out.write(denosing_frame)


    out.release() 
    cap.release()
    cv2.destroyAllWindows()

    return './output_video/cache.mp4'
    


demo = gr.Interface(read_video,
                     gr.Video(),
                    'playable_video',
                    examples=video_list,
                    )
if __name__ =="__main__":
    demo.launch(debug=True, share=False)

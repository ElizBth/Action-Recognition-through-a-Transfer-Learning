from argparse import ArgumentParser

import cv2 as cv
import numpy as np
import os
from decimal import Decimal
from mmpose.apis import MMPoseInferencer
from mmcv.image import imread
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default="/home/elizabeth/Downloads/persons.jpg")
    parser.add_argument('--config', default='/home/elizabeth/Downloads/mmcv/mmdetection/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py')
    parser.add_argument('--checkpoint', default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
    parser.add_argument('--out-file', default='B2.jpg')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--draw-heatmap',default=False)
    parser.add_argument('--show-kpt-idx',default=False)
    parser.add_argument('--skeleton-style',default='mmpose')
    parser.add_argument('--kpt-thr',default=0.5)
    parser.add_argument('--radius',default=5)
    parser.add_argument('--thickness', default=2)
    parser.add_argument('--alpha',default=0.8)
    parser.add_argument('--show',default=False)    
    parser.add_argument('--output',default='')
    
    args = parser.parse_args()

    return args

def pose_inference(args, model, img):
    # inference a single image
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    ## show the results
    #img = imread(args.img, channel_order='rgb')
    #img = np.zeros((img.shape), dtype = np.uint8)
    
    
    
    img_out = visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        wait_time=0,
        out_file=args.output)
    
    return img_out

args = get_args()

# build the model from a config file and a checkpoint file
if args.draw_heatmap:
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
else:
    cfg_options = None

model = init_model(
    args.config,
    args.checkpoint,
    device=args.device,
    cfg_options=cfg_options)

# init visualizer
model.cfg.visualizer.radius = args.radius
model.cfg.visualizer.alpha = args.alpha
model.cfg.visualizer.line_width = args.thickness
model.cfg.visualizer.save_dir = '../output'

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(
    model.dataset_meta, skeleton_style=args.skeleton_style)


# The video feed is read in as
# a VideoCapture object
cap = cv.VideoCapture("/home/elizabeth/Videos/caminando.mp4")
output_f = '/home/elizabeth/Videos/opflow'
output = '/home/elizabeth/Videos/flowpose'
# ret = a boolean return value from
# getting the frame, first_frame = the
# first frame in the entire video sequence

inferencer = MMPoseInferencer('human')

ret, first_frame = cap.read()

# Converts frame to grayscale because we
# only need the luminance channel for
# detecting edges - less computationally
# expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# Creates an image filled with zero
# intensities with the same dimensions
# as the frame
mask = np.zeros_like(first_frame)

# Sets image saturation to maximum
mask[..., 1] = 255

frame_id = 0

height, width, layers = first_frame.shape  
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv.VideoWriter(os.path.join(output_f, 'flow_run_043.mp4'), fourcc, 30, (width, height))


while(cap.isOpened()):
	
    # ret = a boolean return value from getting
    # the frame, frame = the current frame being
    # projected in the video
    ret, frame = cap.read()
	
    if not ret:
        break
    # Opens a new window and displays the input
    # frame
    #cv.imshow("input", frame)
    
    # Converts each frame to grayscale - we previously
    # only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    
    # Calculates dense optical flow by Farneback method
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                    None,
                                    0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    
    
    args.img = frame
    args.output = os.path.join(output_f, str(f'{frame_id:05d}.jpg'))
    
    pose_img = pose_inference(args, model, rgb)
    
    
    # Opens a new window and displays the output frame
    #cv.imshow("dense optical flow", pose_img)

    #cv.imwrite(os.path.join(output, '{:05d}.jpg'.format(frame_id)), pose_img)
    
    video.write(pose_img)
    # Updates previous frame
    prev_gray = gray
    
                
    frame_id += 1
    # Frames are read by intervals of 1 millisecond. The
    # programs breaks out of the while loop when the
    # user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
   

video.release()

#output_video_path = os.path.join(output, 'out_vid_043.mp4')


#files = [file.replace('.jpg', '') for file in os.listdir(output)]
#height, width, layers = cv.imread(os.path.join(output, files[0]+'.jpg')).shape  
#fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
#video = cv.VideoWriter(output_video_path, fourcc, 30, (width, height))
#files.sort(key=Decimal)
#for file in files:
#    video.write(cv.imread(os.path.join(self.save_dir, file+'.jpg')))

#video.release()
                
                
# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()

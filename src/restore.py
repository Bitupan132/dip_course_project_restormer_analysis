## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##--------------------------------------------------------------
##------- Demo file to test Restormer on your own images---------
## Example usage on directory containing several images:   python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
## Example usage with tile option on a large image: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/' --tile 720 --tile_overlap 32
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
from pdb import set_trace as stx
import numpy as np
import configs

parser = argparse.ArgumentParser(description='Test Restormer on your own images')
parser.add_argument('--input_dir', default=configs.DEGRADED_IMAGES_DIR, type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default=configs.RESTORED_IMAGES_DIR, type=str, help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Motion_Deblurring','Gaussian_Color_Denoising'])
parser.add_argument('--sigma', default=None, type=str, help='Sigma value for non-blind denoising.')

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def get_weights_and_parameters(task, parameters, sigma = None):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Gaussian_Color_Denoising':
        if sigma is not None:
            weights = os.path.join('Denoising', 'pretrained_models', f'gaussian_color_denoising_sigma{sigma}.pth')
        else:
            weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

task    = args.task
inp_dir = args.input_dir
out_dir = args.result_dir
sigma = args.sigma

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
    files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
weights, parameters = get_weights_and_parameters(task, parameters, sigma)

load_arch = run_path(os.path.join('restormer', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")

with torch.no_grad():
    for file_ in tqdm(files):
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = load_img(file_)

        input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

        # Pad the input if not_multiple_of 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model(input_)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:height,:width]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(os.path.split(file_)[-1])[0]

        save_img((os.path.join(out_dir, f+'.png')), restored)

    print(f"\nRestored images are saved at {out_dir}")

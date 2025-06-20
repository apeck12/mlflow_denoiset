import mlflow
import numpy as np
import torch
import itertools
from .model import UNet3d


def get_bounds_one_axis(dim: int, length:int, padding:int) -> list:
    """
    Determine the boundaries to achieve the desired crop
    length with overlap.

    Parameters
    ----------
    dim: max number of pixels along axis
    length: subvolume side length before padding
    padding: number of pixels for padding/overlap

    Returns
    -------
    pts: list of tuples of (start,end) pixels to crop
    """
    pts = [0]
    while pts[-1] < dim:
        if len(pts)%2!=0:
            pts.append(pts[-1]+length+padding)
        else:
            pts.append(pts[-1]-padding)
    if pts[-1] > dim:
        pts[-1] = dim
    pts = [(pts[i], pts[i+1]) for i in np.arange(len(pts))[::2]]
    return pts


def get_bounds_3d(shape: tuple, length: int, padding: int) -> tuple:
    """
    Get boundaries for extracting padded subvolumes from a volume, 
    removing the padding (after denoising for example), and then 
    tiling a new volume with the processed subvolumes. Output arrays
    have shape [number of subvolumes, 6], where each entry corresponds
    to a [xstart,xend,ystart,yend,zstart,zend] subvolume coordinates.
    
    Parameters
    ----------
    shape: volume shape
    length: subvolume side length before padding
    padding: number of pixels for padding/overlap
    
    Returns
    -------
    ibounds: initial cropping bounds for cropping padded subvolumes
    rbounds: filling bounds for where to place the cropped subvolume
    sbounds: subvolume cropping bounds for eliminating padded region
    """
    # enforce even padding
    if padding % 2 != 0:
        padding -= 1
    hpadding = int(padding/2)
    
    # determine bounds for extracting padded subvolumes
    xpts = get_bounds_one_axis(shape[0], length, padding)
    ypts = get_bounds_one_axis(shape[1], length, padding)
    zpts = get_bounds_one_axis(shape[2], length, padding)
    
    ibounds = np.array(list(itertools.product(xpts, ypts, zpts)))
    ibounds = ibounds.reshape(ibounds.shape[0],6)

    # determine bounds for stitching subvolumes into volume
    hpadding = int(padding/2)
    rbounds = ibounds.copy()
    rbounds[:,::2] += hpadding
    rbounds[:,1::2] -= hpadding
    rbounds[ibounds==0] = 0
    rbounds[:,:2][ibounds[:,:2]==shape[0]] = shape[0]
    rbounds[:,2:4][ibounds[:,2:4]==shape[1]] = shape[1]
    rbounds[:,4:][ibounds[:,4:]==shape[2]] = shape[2]

    # determine bounds for cropping subvolumes, accounting for padding
    sbounds = rbounds - ibounds
    sbounds[:,1][sbounds[:,1]==0] = rbounds[sbounds[:,1]==0][:,1]-rbounds[sbounds[:,1]==0][:,0] + sbounds[sbounds[:,1]==0][:,0]
    sbounds[:,3][sbounds[:,3]==0] = rbounds[sbounds[:,3]==0][:,3]-rbounds[sbounds[:,3]==0][:,2] + sbounds[sbounds[:,3]==0][:,2]
    sbounds[:,5][sbounds[:,5]==0] = rbounds[sbounds[:,5]==0][:,5]-rbounds[sbounds[:,5]==0][:,4] + sbounds[sbounds[:,5]==0][:,4]

    return ibounds, rbounds, sbounds


class DenoiseMLFLowModel(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """
        Load the model with pre-trained weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet3d()
        self.model.load_state_dict(torch.load(context.artifacts["pytorch_model"]))
        self.model.to(self.device)
        self.model.eval()  
        self.length = 128
        self.padding = 24

    def predict(self, context, model_input: np.ndarray) -> np.ndarray:
        """
        Perform inference -- denoise an input volume. Due to memory
        constraints, denoising is performed patchwise on overlapping
        subvolumes. Borders are discarded to reduce artifacts.
        """
        ibounds, rbounds, sbounds = get_bounds_3d(model_input.shape, self.length, self.padding)
        while np.min(ibounds[:,1::2] - ibounds[:,::2]) < 32:
            padding += 2
            ibounds, rbounds, sbounds = get_bounds_3d(model_input.shape, self.length, self.padding)
        
        volume = torch.from_numpy(model_input)
        volume = volume.to(self.device)
        volume_d = torch.zeros_like(volume)
        volume = volume.unsqueeze(0).unsqueeze(0)
        mu, sigma = volume.mean(), volume.std()
        volume = (volume - mu) / sigma

        for i in range(ibounds.shape[0]):
            volume_i = volume[:,:,ibounds[i][0]:ibounds[i][1],ibounds[i][2]:ibounds[i][3],ibounds[i][4]:ibounds[i][5]]
            with torch.no_grad():
                volume_i = self.model(volume_i).squeeze()
            volume_i = volume_i[sbounds[i][0]:sbounds[i][1],sbounds[i][2]:sbounds[i][3],sbounds[i][4]:sbounds[i][5]]
            volume_d[rbounds[i][0]:rbounds[i][1],rbounds[i][2]:rbounds[i][3],rbounds[i][4]:rbounds[i][5]] = volume_i        

        volume_d = sigma * volume_d + mu
        volume_d = volume_d.cpu().numpy()

        return volume_d

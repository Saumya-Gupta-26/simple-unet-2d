import numpy as np
import os
import cc3d
from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import os, glob
from PIL import Image

srcdir = "/data/saumgupta/simple-unet-2d/inference-output/topo"
logfile = os.path.join(srcdir,"metrics.csv") # will log to this file

def main():

    filepaths = glob.glob(srcdir + "/pred*png")
    filepaths.sort()

    print("Saving results to {}".format(logfile))

    metrics = {'dice':[], 'cldice':[], '0betti':[]}
    with open(logfile, 'a') as wfile:
        for i, fpath in enumerate(filepaths):
            writestr = fpath.split('/')[-1]
            gtpath = fpath.replace("pred","gt")

            pred = interpolate(np.array(Image.open(fpath)))
            pred = np.squeeze(pred)

            gt = interpolate(np.array(Image.open(gtpath)))
            gt = np.squeeze(gt[:,:,0])

            cldice_acc = clDice(pred, gt)
            betti_acc = get_betti_error(pred, gt)
            dice_acc = dice_score(pred, gt)

            metrics['dice'].append(dice_acc)
            metrics['cldice'].append(cldice_acc)
            metrics['0betti'].append(betti_acc)

            writestr += "; Dice {}; clDice {}; 0-dim Betti {}\n".format(dice_acc, cldice_acc, betti_acc)

            wfile.write(writestr)

        wfile.write("\nAverage:\n")
        for key, val in metrics.items():
            avg = np.array(val).mean()
            wfile.write("{}: {}\n".format(key, avg))

# logic okay for binary images 
def interpolate(arr):
    return arr/np.max(arr)

def dice_score(image1, image2):
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Calculate the intersection and the sum of white pixels in each image
    intersection = np.sum(image1_flat * image2_flat)
    sum_values = np.sum(image1_flat) + np.sum(image2_flat)

    # Compute the Dice score
    dice = 2.0 * intersection / sum_values

    return dice

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def conn_comp(arr):
    labels_out, numcomp = cc3d.connected_components(arr, connectivity=26, return_N=True) # 26-connected
    return numcomp

def get_betti_error(arr1, arr2, patchsize=[64,64], stepsize=[64,64]):
    arrsize = arr1.shape
    all_betti = []
    
    for x in range(0,arrsize[0],stepsize[0]):
        for y in range(0,arrsize[1],stepsize[1]):
            newidx = [x+patchsize[0],y+patchsize[1]]
            if(check_bounds([x,y],arrsize) and check_bounds(newidx,arrsize)):
                minivol1 = arr1[x:newidx[0],y:newidx[1]]
                minians1 = conn_comp(minivol1)

                minivol2 = arr2[x:newidx[0],y:newidx[1]]
                minians2 = conn_comp(minivol2)

                all_betti.append(np.abs(minians1-minians2))

    avg_betti = np.asarray(all_betti).mean()
    return avg_betti

def check_bounds(idx, volsize):
    if idx[0] < 0 or idx[0] > volsize[0]:
        return False
    if idx[1] < 0 or idx[1] > volsize[1]:
        return False
    return True

if __name__ == "__main__":
    main()
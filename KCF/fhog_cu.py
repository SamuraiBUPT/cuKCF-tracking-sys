import numpy as np
import cv2
from numba import jit
import torch
import cuKCF

# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getFeatureMaps(image, k, mapp):
    kernel = np.array([[-1., 0., 1.]], np.float32)

    height = image.shape[0]
    width = image.shape[1]
    assert (image.ndim == 3 and image.shape[2])
    numChannels = 3  # (1 if image.ndim==2 else image.shape[2])

    sizeX = width // k
    sizeY = height // k
    px = 3 * NUM_SECTOR
    p = px
    stringSize = sizeX * p

    # the only difference
    if sizeX < 3 or sizeY < 3:
        sizeX += 1
        sizeY += 1

    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['numFeatures'] = p
    mapp['map'] = torch.zeros(
        (mapp['sizeX'] * mapp['sizeY'] * mapp['numFeatures']), dtype=torch.float32, device=device)

    # np.float32(...) is necessary
    dx = torch.tensor(cv2.filter2D(np.float32(image), -1, kernel), device=device)
    dy = torch.tensor(cv2.filter2D(np.float32(image), -1, kernel.T), device=device)

    arg_vector = np.arange(
        NUM_SECTOR + 1).astype(np.float32) * np.pi / NUM_SECTOR
    boundary_x = torch.tensor(np.cos(arg_vector), device=device)
    boundary_y = torch.tensor(np.sin(arg_vector), device=device)

    r2 = torch.zeros((height, width), dtype=torch.float32, device=device)
    alfa2 = torch.zeros((height, width, 2), dtype=torch.int32, device=device)
    
    cuKCF.launch_func1(r2, alfa2, dx, dy, boundary_x,
                       boundary_y, height, width, numChannels, NUM_SECTOR)
    # func2
    nearest = torch.tensor(np.ones((k), np.int32), device=device)
    nearest[0:k // 2] = -1

    w = np.zeros((k, 2), np.float32)
    a_x = np.concatenate((k / 2 - np.arange(k / 2) - 0.5,
                         np.arange(k / 2, k) - k / 2 + 0.5)).astype(np.float32)
    b_x = np.concatenate((k / 2 + np.arange(k / 2) + 0.5, -
                         np.arange(k / 2, k) + k / 2 - 0.5 + k)).astype(np.float32)
    w[:, 0] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
    w[:, 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))

    w = torch.tensor(w, device=device)
    map2 = torch.zeros(mapp['map'].shape, dtype=torch.float32, device=device)
    
    cuKCF.launch_func2(map2, r2, alfa2, nearest, w, k, height,
                       width, sizeX, sizeY, p, stringSize, NUM_SECTOR)
    mapp['map'] = map2

    return mapp


def normalizeAndTruncate(mapp, alfa):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = NUM_SECTOR
    xp = NUM_SECTOR * 3
    pp = NUM_SECTOR * 12

    idx = np.arange(0, sizeX * sizeY * mapp['numFeatures'],
                    mapp['numFeatures']).reshape((sizeX * sizeY, 1)) + np.arange(p)
    np_tmp = mapp['map'].cpu().numpy()
    partOfNorm = torch.tensor(np.sum(np_tmp[idx] ** 2, axis=1), device=device)  # ~0.0002s

    sizeX, sizeY = sizeX - 2, sizeY - 2
    # 30x speedup
    new_data3 = torch.zeros((sizeY * sizeX * pp), dtype=torch.float32, device=device)
    # map_data = torch.tensor(mapp['map'], device=device)
    cuKCF.launch_func3(new_data3, partOfNorm, mapp['map'], sizeX,
                       sizeY, p, xp, pp, partOfNorm.shape[0], mapp['map'].shape[0])
    newData = new_data3
    # truncation
    newData[newData > alfa] = alfa

    mapp['numFeatures'] = pp
    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['map'] = newData

    return mapp


def PCAFeatureMaps(mapp):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = mapp['numFeatures']
    pp = NUM_SECTOR * 3 + 4
    yp = 4
    xp = NUM_SECTOR

    nx = 1.0 / np.sqrt(xp * 2)
    ny = 1.0 / np.sqrt(yp)
    # 190x speedup
    
    new_data3 = torch.zeros((sizeX * sizeY * pp), dtype=torch.float32, device=device)
    # map_data = torch.tensor(mapp['map'], device=device)
    cuKCF.launch_func4(new_data3, mapp['map'], p, sizeX, sizeY,
                       pp, yp, xp, nx, ny, mapp['map'].shape[0])
    newData = new_data3
    mapp['numFeatures'] = pp
    mapp['map'] = newData.cpu().numpy()

    return mapp

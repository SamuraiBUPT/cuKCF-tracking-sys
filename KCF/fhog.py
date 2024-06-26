import numpy as np
import cv2
from numba import jit
# import cuKCF

# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07


@jit
def hp_op1(dx, dy, boundary_x, boundary_y, height, width, numChannels):
    r = np.zeros((height, width), np.float32)
    alfa = np.zeros((height, width, 2), np.int32)

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            c = 0
            x = dx[j, i, c]
            y = dy[j, i, c]
            r[j, i] = np.sqrt(x * x + y * y)

            for ch in range(1, numChannels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = np.sqrt(tx * tx + ty * ty)
                if (magnitude > r[j, i]):
                    r[j, i] = magnitude
                    c = ch
                    x = tx
                    y = ty

            mmax = boundary_x[0] * x + boundary_y[0] * y
            maxi = 0

            for kk in range(0, NUM_SECTOR):
                dotProd = boundary_x[kk] * x + boundary_y[kk] * y
                if (dotProd > mmax):
                    mmax = dotProd
                    maxi = kk
                elif (-dotProd > mmax):
                    mmax = -dotProd
                    maxi = kk + NUM_SECTOR

            alfa[j, i, 0] = maxi % NUM_SECTOR
            alfa[j, i, 1] = maxi
    return r, alfa


@jit
def hp_op2(dx, dy, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize):
    mapp = np.zeros((sizeX * sizeY * p), np.float32)
    for i in range(sizeY):
        for j in range(sizeX):
            for ii in range(k):
                for jj in range(k):
                    if ((i * k + ii > 0) and (i * k + ii < height - 1) and (j * k + jj > 0) and (j * k + jj < width - 1)):
                        mapp[i * stringSize + j * p + alfa[k*i+ii, j*k+jj, 0]
                             ] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]  # ii0 jj0
                        mapp[i * stringSize + j * p + alfa[k*i+ii, j*k+jj, 1] +
                             NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]    # ii0 jj0
                        if ((i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1)):
                            mapp[(i + nearest[ii]) * stringSize + j * p + alfa[k*i+ii, j*k+jj, 0]
                                 ] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]  # ii1 jj0
                            mapp[(i + nearest[ii]) * stringSize + j * p + alfa[k*i+ii, j*k+jj, 1] +
                                 NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]    # ii1 jj0
                        if ((j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1)):
                            mapp[i * stringSize + (j + nearest[jj]) * p + alfa[k*i+ii, j*k+jj, 0]
                                 ] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 1]  # ii0 jj1
                            mapp[i * stringSize + (j + nearest[jj]) * p + alfa[k*i+ii, j*k+jj, 1] +
                                 NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 1]    # ii0 jj1
                        if ((i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1) and (j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1)):
                            mapp[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa[k*i+ii,
                                                                                               j*k+jj, 0]] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]  # ii1 jj1
                            mapp[(i + nearest[ii]) * stringSize + (j + nearest[jj]) * p + alfa[k*i+ii, j*k+jj,
                                                                                               1] + NUM_SECTOR] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]    # ii1 jj1
    return mapp


@jit
def hp_op3(partOfNorm, mappmap, sizeX, sizeY, p, xp, pp):
    newData = np.zeros((sizeY * sizeX * pp), np.float32)
    for i in range(1, sizeY + 1):
        for j in range(1, sizeX + 1):
            pos1 = i * (sizeX + 2) * xp + j * xp
            pos2 = (i - 1) * sizeX * pp + (j - 1) * pp

            valOfNorm = np.sqrt(partOfNorm[(i) * (sizeX + 2) + (j)] +
                                partOfNorm[(i) * (sizeX + 2) + (j + 1)] +
                                partOfNorm[(i + 1) * (sizeX + 2) + (j)] +
                                partOfNorm[(i + 1) * (sizeX + 2) + (j + 1)]) + FLT_EPSILON
            newData[pos2:pos2 + p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 4 * p:pos2 + 6 *
                    p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            valOfNorm = np.sqrt(partOfNorm[(i) * (sizeX + 2) + (j)] +
                                partOfNorm[(i) * (sizeX + 2) + (j + 1)] +
                                partOfNorm[(i - 1) * (sizeX + 2) + (j)] +
                                partOfNorm[(i - 1) * (sizeX + 2) + (j + 1)]) + FLT_EPSILON
            newData[pos2 + p:pos2 + 2 * p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 6 * p:pos2 + 8 *
                    p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            valOfNorm = np.sqrt(partOfNorm[(i) * (sizeX + 2) + (j)] +
                                partOfNorm[(i) * (sizeX + 2) + (j - 1)] +
                                partOfNorm[(i + 1) * (sizeX + 2) + (j)] +
                                partOfNorm[(i + 1) * (sizeX + 2) + (j - 1)]) + FLT_EPSILON
            newData[pos2 + 2 * p:pos2 + 3 *
                    p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 8 * p:pos2 + 10 *
                    p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm

            valOfNorm = np.sqrt(partOfNorm[(i) * (sizeX + 2) + (j)] +
                                partOfNorm[(i) * (sizeX + 2) + (j - 1)] +
                                partOfNorm[(i - 1) * (sizeX + 2) + (j)] +
                                partOfNorm[(i - 1) * (sizeX + 2) + (j - 1)]) + FLT_EPSILON
            newData[pos2 + 3 * p:pos2 + 4 *
                    p] = mappmap[pos1:pos1 + p] / valOfNorm
            newData[pos2 + 10 * p:pos2 + 12 *
                    p] = mappmap[pos1 + p:pos1 + 3 * p] / valOfNorm
    return newData


@jit
def hp_op4(mappmap, p, sizeX, sizeY, pp, yp, xp, nx, ny):
    newData = np.zeros((sizeX * sizeY * pp), np.float32)
    for i in range(sizeY):
        for j in range(sizeX):
            pos1 = (i * sizeX + j) * p
            pos2 = (i * sizeX + j) * pp

            for jj in range(2 * xp):  # 2*9
                newData[pos2 + jj] = np.sum(mappmap[pos1 + yp *
                                            xp + jj: pos1 + 3 * yp * xp + jj: 2 * xp]) * ny
            for jj in range(xp):  # 9
                newData[pos2 + 2 * xp +
                        jj] = np.sum(mappmap[pos1 + jj: pos1 + jj + yp * xp: xp]) * ny
            for ii in range(yp):  # 4
                newData[pos2 + 3 * xp + ii] = np.sum(
                    mappmap[pos1 + yp * xp + ii * xp * 2: pos1 + yp * xp + ii * xp * 2 + 2 * xp]) * nx
    return newData


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
    mapp['map'] = np.zeros(
        (mapp['sizeX'] * mapp['sizeY'] * mapp['numFeatures']), np.float32)

    # np.float32(...) is necessary
    dx = cv2.filter2D(np.float32(image), -1, kernel)
    dy = cv2.filter2D(np.float32(image), -1, kernel.T)

    arg_vector = np.arange(
        NUM_SECTOR + 1).astype(np.float32) * np.pi / NUM_SECTOR
    boundary_x = np.cos(arg_vector)
    boundary_y = np.sin(arg_vector)

    # 200x speedup
    r, alfa = hp_op1(dx, dy, boundary_x, boundary_y, height, width,
                    numChannels)  # modify: I change alfa to 1-D array

    # verify_hp_op1(r, alfa, dx, dy, boundary_x, boundary_y, height, width, numChannels)  # test zone
    # ~0.001s

    # hp_op2
    nearest = np.ones((k), np.int32)
    nearest[0:k // 2] = -1

    w = np.zeros((k, 2), np.float32)
    a_x = np.concatenate((k / 2 - np.arange(k / 2) - 0.5,
                         np.arange(k / 2, k) - k / 2 + 0.5)).astype(np.float32)
    b_x = np.concatenate((k / 2 + np.arange(k / 2) + 0.5, -
                         np.arange(k / 2, k) + k / 2 - 0.5 + k)).astype(np.float32)
    w[:, 0] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
    w[:, 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))

    '''
    ### original implementation
    mapp['map'] = hp_op2(dx, dy, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize) #hp_op2 without @jit  ###
    '''
    # 500x speedup
    mapp['map'] = hp_op2(dx, dy, boundary_x, boundary_y, r, alfa, nearest,
                        w, k, height, width, sizeX, sizeY, p, stringSize)  # with @jit

    # notice: we will just use the data above, to make a cuda op
    # verify_hp_op2(mapp, dx, dy, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize)  # test zone
    # ~0.001s

    return mapp


def normalizeAndTruncate(mapp, alfa):
    sizeX = mapp['sizeX']
    sizeY = mapp['sizeY']

    p = NUM_SECTOR
    xp = NUM_SECTOR * 3
    pp = NUM_SECTOR * 12

    '''
    ### original implementation
    partOfNorm = np.zeros((sizeY*sizeX), np.float32)

    for i in range(sizeX*sizeY):
        pos = i * mapp['numFeatures']
        partOfNorm[i] = np.sum(mapp['map'][pos:pos+p]**2) ###
    '''
    # 50x speedup
    idx = np.arange(0, sizeX * sizeY * mapp['numFeatures'],
                    mapp['numFeatures']).reshape((sizeX * sizeY, 1)) + np.arange(p)
    partOfNorm = np.sum(mapp['map'][idx] ** 2, axis=1)  # ~0.0002s

    sizeX, sizeY = sizeX - 2, sizeY - 2
    # 30x speedup
    newData = hp_op3(partOfNorm, mapp['map'],
                    sizeX, sizeY, p, xp, pp)  # with @jit

    # verify_hp_op3(newData, partOfNorm, mapp['map'], sizeX, sizeY, p, xp, pp)  # test zone

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
    newData = hp_op4(mapp['map'], p, sizeX, sizeY,
                    pp, yp, xp, nx, ny) 

    mapp['numFeatures'] = pp
    mapp['map'] = newData

    return mapp

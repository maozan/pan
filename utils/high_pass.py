import cv2 as cv
import numpy as np

def ms_process(ms, a, b):
    # print('ms', ms.shape)
    bands = ms.shape[0]
    ms_hp = []
    for i in range(bands):
        ms_i = np.expand_dims(ms[i, :, :], axis=0)
        # print(ms_i.shape)
        f = np.fft.fft2(ms_i)
        fshift = np.fft.fftshift(f)
        #设置高通滤波器
        rows, cols = ms_i.shape[1], ms_i.shape[2]
        crow,ccol = int(rows/2), int(cols/2)
        fshift[crow-a:crow+a, ccol-b:ccol+b] = 0
        #傅里叶逆变换
        ishift = np.fft.ifftshift(fshift)
        ms_i_hp = np.fft.ifft2(ishift)
        ms_i_hp = np.abs(ms_i_hp)
        ms_hp.append(ms_i_hp)
    return np.concatenate(ms_hp, axis=0)

def highPass(pans, mss, a = 30, b = 30):

    ## pan process
    for pan in pans:
        # print(pan.shape)
        f = np.fft.fft2(pan)
        fshift = np.fft.fftshift(f)
        #设置高通滤波器
        rows, cols = pan.shape[1], pan.shape[2]
        crow,ccol = int(rows/2), int(cols/2)
        fshift[crow-a:crow+a, ccol-b:ccol+b] = 0
        #傅里叶逆变换
        ishift = np.fft.ifftshift(fshift)
        pan_hp = np.fft.ifft2(ishift)
        pan_hp = np.abs(pan_hp)

    ## ms process
    for ms in mss:
        ms_hp = ms_process(ms, a, b)

    return ms_hp, pan_hp 
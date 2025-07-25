# import numpy as np
# import cv2
# import math
# import random
# import time
# # import rotate_brush as rb
# # import gradient
# # from thready import amap
# import os

# color1 = np.array([19,163,255.])
# color2 = np.array([19,18,37.])
# color3 = np.array([52,2,1.])

# color4 = np.array([.95,.95,.95])
# color5 = np.array([.5,.15,.4])

# color1/=255.
# color2/=255.
# color3/=255.

# def clip(i,imax=1,imin=0):
#     return min(max(imin,i),imax)

# def directmix(c1,c2,ratio):
#     return c1*ratio + c2*(1-ratio)

# def linearmix(c1,c2,ratio):
#     c1 = pow(c1,2.2)
#     c2 = pow(c2,2.2)
#     mix = c1*ratio + c2*(1-ratio)
#     return pow(mix,1/2.2)

# def submix_power(i):
#     i = np.array(i)

#     def BGR2PWR(c):
#         c = np.clip(c,a_max=1.-1e-6,a_min=1e-6) # no overflow allowed
#         c = np.power(c,2.2/i)
#         u = 1. - c
#         return u # unit absorbance

#     def PWR2BGR(u):
#         c = 1. - u
#         c = np.power(c,i/2.2)
#         return c # rgb color

#     def submix(c1,c2,ratio):
#         uabs1,uabs2 = BGR2PWR(c1),BGR2PWR(c2)
#         mixuabs = (uabs1 * ratio) + (uabs2*(1-ratio))
#         return PWR2BGR(mixuabs)

#     return submix,BGR2PWR,PWR2BGR


# # a is good around .15
# def reflectance(coeff):
#     # reflectance model
#     # receive 1 absorb b pass thru a
#     a,b = coeff[:,0:1],coeff[:,1:2]
#     # print(a.shape,b.shape)

#     return -(a**2 *(a + b - 1)**2 * ( - 1))/((a**2 - 1)* (-a - b + 1)* (a**2 + 2 *a* b - 2* a + b**2 - 2* b)) + (1-a-b);

# def get_coeff(refl):
#     coeff = np.zeros((refl.shape[0],2))
#     refl = np.reshape(refl,refl.shape+(1,))

#     coeff[:,0] += .3
#     coeff[:,1] += .3

#     # a = 0.1
#     # b =.5

#     delta = 1e-5

#     def err(refl,coeff):
#         r = reflectance(coeff)
#         diff = r - refl
#         print(r.shape,refl.shape,coeff.shape,diff.shape)
#         return diff*diff

#     # gd
#     for i in range(8):
#         grad = (err(refl,coeff+delta)-err(refl,coeff))/delta
#         coeff = coeff - grad*.17

#         coeff[:,1] = np.clip(coeff[:,1],a_max=0.9,a_min=0.05)
#         # b: absorbance

#         coeff[:,0] = np.clip(coeff[:,0],a_max=0.1,a_min=0.02)
#         # a = clip(a,imax=0.12,imin=0.05)

#         print('iter {}, coeff:'.format(i),coeff)

#     return coeff


# def powermix(c1,c2,ratio):
#     c1 = pow(c1,2.2)
#     c2 = pow(c2,2.2)

#     coef1,coef2 = get_coeff(c1),get_coeff(c2)
#     mixcoef = coef1*ratio + coef2*(1-ratio)
#     refl = reflectance(mixcoef)

#     return pow(refl,1/2.2)

# pal = np.zeros([400,600,3])

# globpad = 0
# def pmix(f,c1,c2):
#     global globpad
#     pat= 15
#     for i in range(pat):
#         pal[i*20:i*20+18,globpad:globpad+20] = np.reshape(f(c1,c2,clip(1-i/(pat-1))),(-1,3))
#     globpad+=23

# def demo(f):
#     pmix(f,color4,color1)
#     pmix(f,color1,color3)
#     pmix(f,color4,color3)
#     pmix(f,color4,color2)
#     pmix(f,color4,color5)
#     global globpad
#     globpad+=4

# def test_power():
#     for i in range(20):
#         [[a,b]] = get_coeff(np.array([i/20.]))
#         print('input{},coeff a,b:{:5f},{:5f}'.format(i/20,a,b))

# def test():
#     demo(directmix)
#     # demo(linearmix)
#     # demo(submix_power([3.]))
#     # demo(submix_power([2,3,4.]))
#     demo(submix_power([7,1,9.]))
#     # demo(submix_power([6,1,4.]))
#     demo(submix_power([10,2,4.]))
#     demo(submix_power([13,3,7.]))

#     cv2.imshow('palette',pal)
#     cv2.waitKey(0)

# def f32(a):
#     return np.array(a).astype('float32')

# def oilpaint_converters():
#     submix,b2p,p2b = submix_power(f32([13,3,7.]))
#     return b2p,p2b

# def oilpaint_mix(c1,c2,alpha):
#     submix,b2p,p2b = submix_power(f32([13,3,7.]))
#     return submix(c1,c2,alpha/255)

# # test()
# colormixer.py

import numpy as np
import cv2
import math
import random
import time
import os

# ─────────── Example base colors ───────────────────────────────────────
color1 = np.array([19, 163, 255.])
color2 = np.array([19, 18, 37.])
color3 = np.array([52, 2, 1.])

color4 = np.array([.95, .95, .95])
color5 = np.array([.5, .15, .4])

color1 /= 255.
color2 /= 255.
color3 /= 255.

# ─────────── Helpers ───────────────────────────────────────────────────
def clip(i, imax=1, imin=0):
    return min(max(imin, i), imax)

def directmix(c1, c2, ratio):
    return c1 * ratio + c2 * (1 - ratio)

def linearmix(c1, c2, ratio):
    c1_lin = np.power(c1, 2.2)
    c2_lin = np.power(c2, 2.2)
    mix_lin = c1_lin * ratio + c2_lin * (1 - ratio)
    return np.power(mix_lin, 1 / 2.2)

def submix_power(i):
    i = np.array(i)
    def BGR2PWR(c):
        c = np.clip(c, a_max=1. - 1e-6, a_min=1e-6)
        return 1. - np.power(c, 2.2 / i)
    def PWR2BGR(u):
        return np.power(1. - u, i / 2.2)
    def submix(c1, c2, ratio):
        u1 = BGR2PWR(c1)
        u2 = BGR2PWR(c2)
        mixu = u1 * ratio + u2 * (1 - ratio)
        return PWR2BGR(mixu)
    return submix, BGR2PWR, PWR2BGR

def reflectance(coeff):
    a, b = coeff[:, 0:1], coeff[:, 1:2]
    num = -(a**2 * (a + b - 1)**2 * -1)
    den = (a**2 - 1) * (-a - b + 1) * (a**2 + 2*a*b - 2*a + b**2 - 2*b)
    return num/den + (1 - a - b)

def get_coeff(refl):
    refl = refl.reshape(refl.shape + (1,))
    coeff = np.zeros((refl.shape[0], 2))
    coeff[:, 0] = 0.3
    coeff[:, 1] = 0.3
    delta = 1e-5

    def err(r, c):
        return (reflectance(c) - r)**2

    for _ in range(8):
        grad = (err(refl, coeff + delta) - err(refl, coeff)) / delta
        coeff -= grad * 0.17
        coeff[:, 1] = np.clip(coeff[:, 1], 0.05, 0.9)
        coeff[:, 0] = np.clip(coeff[:, 0], 0.02, 0.1)
    return coeff

def powermix(c1, c2, ratio):
    c1_lin = np.power(c1, 2.2)
    c2_lin = np.power(c2, 2.2)
    coef1 = get_coeff(c1_lin)
    coef2 = get_coeff(c2_lin)
    mixcoef = coef1 * ratio + coef2 * (1 - ratio)
    refl = reflectance(mixcoef)
    return np.power(refl, 1 / 2.2)

# ─────────── Utilities for demo & oil-painting ────────────────────────
pal = np.zeros([400, 600, 3])
globpad = 0

def pmix(f, c1, c2):
    global globpad
    pat = 15
    for i in range(pat):
        row = f(c1, c2, clip(1 - i/(pat-1)))
        pal[i*20:i*20+18, globpad:globpad+20] = row.reshape(-1, 3)
    globpad += 23

def demo(f):
    pmix(f, color4, color1)
    pmix(f, color1, color3)
    pmix(f, color4, color3)
    pmix(f, color4, color2)
    pmix(f, color4, color5)
    global globpad
    globpad += 4

def f32(a):
    return np.array(a, dtype='float32')

def oilpaint_converters():
    submix, b2p, p2b = submix_power(f32([13,3,7.]))
    return b2p, p2b

def oilpaint_mix(c1, c2, alpha):
    submix, _, _ = submix_power(f32([13,3,7.]))
    return submix(c1, c2, alpha/255)

# ─────────── Latent-space API stubs (for Flask “mixbox”) ──────────────

# Here we treat the "latent vector" simply as the 3-channel RGB itself:
LATENT_SIZE = 3

def rgb_to_latent(rgb):
    """
    Identity embed: interpret [r, g, b] as the latent vector.
    """
    return [float(c) for c in rgb]

def latent_to_rgb(latent):
    """
    Identity decode: round/clamp a 3-element vector back to (r, g, b) integers.
    """
    vals = [round(v) for v in latent]
    return (int(vals[0]), int(vals[1]), int(vals[2]))

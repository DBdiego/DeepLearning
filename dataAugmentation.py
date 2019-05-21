import cv2
import numpy as np

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    
    for X_img in X_imgs:
        gaussian = (np.random.random((row, col, 1))*255).astype(np.float32)
        
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, gaussian, 0.35, 0)
        gaussian_noise_imgs.append(gaussian_img)
        
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    
    return gaussian_noise_imgs

'''
X_imgs = [cv2.imread('./database/Alfa Romeo10882_small.jpg', 1).astype(np.float32)]
gaussian_noise_imgs = add_gaussian_noise(X_imgs)
cv2.imwrite('messigray.png',gaussian_noise_imgs[0])
'''

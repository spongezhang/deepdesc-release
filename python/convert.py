import sys
import cv2
import os, sys
import math
import numpy as np
import lutorpy as lua
import scipy.io as sio

class LutorpyNet:
    """
    Wrapper to Torch for loading models
    """
    # TODO: check codes for a bigger batch size.
    # Now it only works with size of one.
    batch_sz = 10000
    # TFeat number of input channels
    input_channels = 1
    # TFeat image input size
    input_sz = 32
    # TFeat descriptor size
    descriptor_sz = 128

    def __init__(self, model_file):
        """
        Class constructor

        :param model_file: The torch file with the trained model
        """
        require('nn')
        require('cudnn')
        self.net = torch.load(model_file)
        self.ones_arr = np.ones((self.input_sz, self.input_sz), dtype=np.uint8)

    def rectify_patch(self, img, kp, patch_sz):
        """
        Extract and rectifies the patch from the original image given a keyopint

        :param img: The input image
        :param kp: The OpenCV keypoint object
        :param patch_sz: The size of the patch to extract

        :return rot: The rectified patch
        """
        # TODO: check this routine since it does not work at all

        scale = 1.0 * float(kp.size) / float(patch_sz)

        c = 1.0 if (kp.angle < 0) else np.cos(kp.angle)
        s = 0.0 if (kp.angle < 0) else np.sin(kp.angle)

        M = np.array([[scale*c, -scale*s, (-scale*c + scale*s) * patch_sz / 2.0 + kp.pt[0]],
                     [scale*s,  scale*c, (-scale*s - scale*c) * patch_sz / 2.0 + kp.pt[1]]])

        rot = cv2.warpAffine(img, np.float32(M), (patch_sz, patch_sz), \
              flags = cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC) #+ cv2.WARP_FILL_OUTLIERS

        return rot

    def extract_patches(self, img, kpts):
        """
        Extract the patches and subtract the mean

        :param img: The input image
        :param kpts: The set of OpenCV keypoint objects
        
        :return: An array with the patches with zero mean
        """
        patches = []
        for kp in kpts:
            # extract patch
            # sub = cv2.getRectSubPix(img, (int(kp.size*1.3), int(kp.size*1.3)), kp.pt)
            sub = self.rectify_patch(img, kp, self.input_sz)
  
            # resize the patch
            res = cv2.resize(sub, (self.input_sz, self.input_sz))
            # subtract mean
            nmean = res - (self.ones_arr * cv2.mean(res)[0])
            nmean = nmean.reshape(self.input_channels, self.input_sz, self.input_sz)
            patches.append(nmean)

        return np.asarray(patches)

    def compute(self, img, kpts):
        """
        Compute the descriptors given a set of keypoints

        :param img: The input image
        :param kpts: The set of OpenCV keypoint objects
        
        :return: An array the descriptors
        """
        # number of keypoints
        N = len(kpts)
        # extract the patches given the keypoints
        patches = self.extract_patches(img, kpts)
        assert N == len(patches)
        print N
        # convert numpy array to torch tensor
        patches_t = torch.fromNumpyArray(patches)
        patches_t._view(N, self.input_channels, self.input_sz, self.input_sz)

        # split patches into batches
        patches_t   = patches_t._split(self.batch_sz)
        descriptors = []

        for i in range(int(np.ceil(float(N) / self.batch_sz))):
           # infere Torch network
            prediction_t = self.net._forward(patches_t[i]._cuda())
           
           # Cast TorchTensor to NumpyArray and append to results
            prediction = prediction_t.asNumpyArray()

           # add the current prediction to the buffer
            descriptors.append(prediction)

        return np.float32(np.asarray(descriptors).reshape(N, self.descriptor_sz))


def main():

    # create CNN descriptor
    torch_file = './nets/tfeat_liberty_ratio_star.t7'
    net = LutorpyNet(torch_file)
    maxsize = 1024*768
    # initialise ORB detector
    working_dir = '/home/xuzhang/project/Medifor/data/'
    #working_dir = '/Users/Xu/program/Image_Genealogy/data/'

    dataset_name = 'NIMBLE2016' 
    subset_name = {'images', 'query'}
    feature_name = 'sift_vlfeat'
    save_feature_name = 'sift_tfeat'
    index = 1;
    
    for subset in subset_name:
        os.system('mkdir -p '+working_dir+dataset_name+'/'+subset+'_'+save_feature_name+'/');
        for im_name in os.listdir(working_dir+dataset_name+'/'+subset):
            if im_name.endswith(".jpg") or im_name.endswith(".JPG") or \
                im_name.endswith(".png") or im_name.endswith(".PNG") or \
                im_name.endswith(".tif") or im_name.endswith(".TIF") or \
                im_name.endswith(".bmp") or im_name.endswith(".BMP"):
                    print im_name
                    index = index+1
                    img = cv2.imread(working_dir+dataset_name+'/'+subset+'_origin/'+im_name)
                    if img.shape[0]*img.shape[1]>maxsize:
                        real_height = int(math.sqrt(float(maxsize)/(img.shape[1]*img.shape[0]))*img.shape[0])
                        real_width = int(math.sqrt(float(maxsize)/(img.shape[1]*img.shape[0]))*img.shape[1])
                        dst_img = np.zeros((real_height,real_width,3), np.uint8)
                        dst_img = cv2.resize(img,(real_width,real_height))
                        img = dst_img
                    keypt = sio.loadmat(working_dir+dataset_name+'/'+subset+'_'+feature_name+'/'+im_name[0:-4]+'.mat')
                    meta = keypt['meta']
                    kp_list = []
                    for i in range(0,meta.shape[1]):
                        kp_temp=cv2.KeyPoint(meta[0,i], meta[1,i], meta[2,i]*16, meta[3,i])
                        kp_list.append(kp_temp)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # # find the keypoints and the descriptors
                    features = net.compute(gray, kp_list)
                    save_object = np.zeros((2,),dtype=np.object)
                    save_object[0] = features
                    save_object[1] = meta
                    sio.savemat(working_dir+dataset_name+'/'+subset+'_'+save_feature_name+'/'+im_name[0:-4]+'.mat', \
                    {'save_object':save_object})
   # read object image and compute descriptors
   # img = cv2.imread(object_img)
   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # # find the keypoints and the descriptors
   # kp12  = det.detect(gray, None)
   # des12 = net.compute(gray, kp12)
   # kp22 = kp12
   # _ , des22 = det.compute(gray, kp12)
    # video loop

if __name__ == '__main__':

    main() 

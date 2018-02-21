% Script to create cropped image and segmentation from ADE20K dataset

clear; close all; clc % clear out environment

%% Create cropped image, segmentation

img      = imread('Scene.jpg');
img_crop = imcrop(img, [990 785 130 205]);
imwrite(img_crop, 'couple.png');

seg      = imread('Annotator1_seg.png');
seg_crop = imcrop(seg, [990 785 130 205]);
imwrite(seg_crop, 'gt_seg.png');

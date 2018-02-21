% Script to create cropped image and segmentation from ADE20K dataset

clear; close all; clc % clear out environment

%% Create cropped image, segmentation

img      = imread('Original_Data/Scene.jpg');
img_crop = imcrop(img, [990 785 130 205]);
imwrite(img_crop, 'Generated/couple.png');

seg      = imread('Original_Data/Annotator1_seg.png');
seg_crop = imcrop(seg, [990 785 130 205]);

% Set black (unlabeled) pixels to background color
% They're all close enough in the image we're using, so this is reasonable
for i=1:size(seg_crop, 1)
    for j=1:size(seg_crop, 2)
        if seg_crop(i, j, 1) == 0 && seg_crop(i, j, 2) == 0 && seg_crop(i, j, 3) == 0
            seg_crop(i,j,1) = 10;
            seg_crop(i,j,2) = 56;
            seg_crop(i,j,3) = 112;
        end
    end
end

imwrite(seg_crop, 'Generated/gt_seg.png');

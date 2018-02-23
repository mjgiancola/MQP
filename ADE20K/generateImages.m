% Script to create cropped image and segmentation from ADE20K dataset

clear; close all; clc % clear out environment

%% Create cropped image, segmentation

img      = imread('Original_Data/Scene.jpg');

img_crop = imcrop(img, [990 785 130 205]);
%imwrite(img_crop, 'Generated/couple/couple.png');

img_crop2 = imcrop(img, [515 625 55 130]);
%imwrite(img_crop2, 'Generated/flag/flag.png');

img_crop3 = imcrop(img, [425 790 50 65]);
%imwrite(img_crop3, 'Generated/people/people.png');

img_crop4 = imcrop(img, [805 450 125 120]);
%imwrite(img_crop4, 'Generated/light/light.png');

seg      = imread('Original_Data/Annotator1_seg.png');
seg2     = imread('Original_Data/Annotator2_seg.png');

% Set black (unlabeled) pixels to background color
% They're all close enough in the images we're using, so this is reasonable

seg_crop = imcrop(seg, [990 785 130 205]);
for i=1:size(seg_crop, 1)
    for j=1:size(seg_crop, 2)
        if seg_crop(i, j, 1) == 0 && seg_crop(i, j, 2) == 0 && seg_crop(i, j, 3) == 0
            seg_crop(i,j,1) = 10;
            seg_crop(i,j,2) = 56;
            seg_crop(i,j,3) = 112;
        end
    end
end
%imwrite(seg_crop, 'Generated/couple/gt_seg.png');

seg_crop2 = imcrop(seg2, [515 625 55 130]);
%imwrite(seg_crop2, 'Generated/flag/gt_seg.png');

seg_crop3 = imcrop(seg2, [425 790 50 65]);
%imwrite(seg_crop3, 'Generated/people/gt_seg.png');

seg_crop4 = imcrop(seg, [805 450 125 120]);
for i=1:size(seg_crop4, 1)
    for j=1:size(seg_crop4, 2)
        if seg_crop4(i, j, 1) == 0 && seg_crop4(i, j, 2) == 0 && seg_crop4(i, j, 3) == 0
            seg_crop4(i,j,1) = 10;
            seg_crop4(i,j,2) = 56;
            seg_crop4(i,j,3) = 92;
        end
    end
end
%imwrite(seg_crop4, 'Generated/light/gt_seg.png');

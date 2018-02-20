% Script to create labels dataset from images in ADE20K dataset
% This generates a small subset used for initial testing purposes

clear; close all; clc % clear out environment

load('index_ade20k.mat');

outfile = fopen('couple.txt', 'w');

%% Read in image data

% These files don't actually exist;
% loadAde20K will parse them to get the other paths
filename1 = 'Annotator1.jpg';
filename2 = 'Annotator2.jpg';
filename3 = 'Annotator3.jpg';
filename4 = 'Annotator4.jpg';
[Om1, Oi1, Pm1, Pi1, objects1, parts1] = loadAde20K(filename1);
[Om2, Oi2, Pm2, Pi2, objects2, parts2] = loadAde20K(filename2);
[Om3, Oi3, Pm3, Pi3, objects3, parts3] = loadAde20K(filename3);
[Om4, Oi4, Pm4, Pi4, objects4, parts4] = loadAde20K(filename4);
% Om is a matrix where Om(i,j) contains the label of pixel i,j

% For first experiment, take a slice of images
Om1 = Om1(785:990, 990:1120);
Om2 = Om2(785:990, 990:1120);
Om3 = Om3(785:990, 990:1120);
Om4 = Om4(785:990, 990:1120);

%% Generate list of unique label types in all images
labels1 = unique(Om1);
labels2 = unique(Om2);
labels3 = unique(Om3);
labels4 = unique(Om4);
labels = unique([labels1; labels2; labels3; labels4]);

%% Optional: Print label types

%for i=2:size(labels)
%    index.objectnames{labels(i)}
%end
%return

%% Compute metadata

% Number of pixels in the image (all are same)
numImages = numel(Om1);

% Each pixel is labeled by each annotator
numLabels = 4 * numImages;

numLabelers = 4;

% Follows notation of our paper
% Number of unique colors in all pictures
numCharacters = size(labels, 1);

prior = 1 / numCharacters;

% Gross, I know. Outputs metadata to file
% Easiest way to include character set is to write manually
% It is stored in a variable but most labels have several names,
% so I looked at it manually and picked one
fprintf(outfile, '%d %d %d %d\r\n', numLabels, numLabelers, numImages, numCharacters);
fprintf(outfile, 'Missing Purse Building Person Road Sidewalk Stoplight Umbrella Pavement\r\n');
fprintf(outfile, '%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\r\n', ...
        prior, prior, prior, prior, prior, prior, prior, prior, prior);

%% Read labels for image 1

labelerIdx = 0;
xdim = size(Om1, 1);
ydim = size(Om1, 2);
    
for i = 1:xdim
    for j = 1:ydim
        pixel = Om1(i,j);
        imageIdx = (ydim * (i-1)) + (j-1);
        % Gives a label from 0,..,<numCharacters>-1 for each pixel
        lbl = find(labels==pixel) - 1;
        fprintf(outfile, '%d %d %d\r\n', imageIdx, labelerIdx, lbl);
    end
end

%% Read labels from image 2

labelerIdx = 1;
    
for i = 1:xdim
    for j = 1:ydim
        pixel = Om2(i,j);
        imageIdx = (ydim * (i-1)) + (j-1);
        % Gives a label from 0,..,<numCharacters>-1 for each pixel
        lbl = find(labels==pixel) - 1;
        fprintf(outfile, '%d %d %d\r\n', imageIdx, labelerIdx, lbl);
    end
end

%% Read labels from image 3

labelerIdx = 2;
    
for i = 1:xdim
    for j = 1:ydim
        pixel = Om3(i,j);
        imageIdx = (ydim * (i-1)) + (j-1);
        % Gives a label from 0,..,<numCharacters>-1 for each pixel
        lbl = find(labels==pixel) - 1;
        fprintf(outfile, '%d %d %d\r\n', imageIdx, labelerIdx, lbl);
    end
end

%% Read labels from image 4

labelerIdx = 3;
    
for i = 1:xdim
    for j = 1:ydim
        pixel = Om4(i,j);
        imageIdx = (ydim * (i-1)) + (j-1);
        % Gives a label from 0,..,<numCharacters>-1 for each pixel
        lbl = find(labels==pixel) - 1;
        fprintf(outfile, '%d %d %d\r\n', imageIdx, labelerIdx, lbl);
    end
end

%% Done !!!
fclose(outfile);
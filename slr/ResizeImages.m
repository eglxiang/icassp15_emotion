function [resized_images] = ResizeImages(images,down_factor)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[num_images,height,width]=size(images);
new_height=floor(height/down_factor);
new_width=floor(width/down_factor);
resized_images=zeros(num_images,new_height,new_width);
for i=1:num_images
    temp_image=reshape(images(i,:,:),[height,width]);
    temp_image=imresize(temp_image,[new_height,new_width],'bicubic');
    temp_image=reshape(temp_image,[1,new_height,new_width]);
    resized_images(i,:,:)=temp_image;
end
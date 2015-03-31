function [resized_images] = ResizeImages2(images,new_height,new_width)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[num_images,height,width]=size(images);
resized_images=zeros(num_images,new_height,new_width);
for i=1:num_images
    temp_image=reshape(images(i,:,:),[height,width]);
    temp_image=imresize(temp_image,[new_height,new_width],'bicubic');
    temp_image=reshape(temp_image,[1,new_height,new_width]);
    resized_images(i,:,:)=temp_image;
end


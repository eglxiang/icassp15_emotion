function [Dictionary]=GenerateDictionary(video_directory,train_indices,video_mode,new_height,new_width,num_frames_per_video,do_normalize)

cd(video_directory);
exterior_folders=ls;
[num_exterior_folders,~]=size(exterior_folders);

[num_classes, train_samples_per_class]=size(train_indices);

Dictionary=zeros(new_height*new_width,num_classes*train_samples_per_class*num_frames_per_video);

position_to_insert=1;

for i=3:num_exterior_folders
    current_exterior_folder=exterior_folders(i,:);
    cd(current_exterior_folder);
    interior_files=ls;
    [num_interior_files,~]=size(interior_files);
    num_videos_current_emotion=(num_interior_files-2)/2;% half, half
    file_indices = train_indices(i-2,:)+ (2 + num_videos_current_emotion);
    for j=1:train_samples_per_class
        video_name =interior_files(file_indices(j),:);    
        load(video_name);
        [column_video_segment]= ProcessVideo(video_matrix,new_height,new_width,video_mode,num_frames_per_video,do_normalize); %Images are now columns of a matrix
        Dictionary(:,position_to_insert:position_to_insert-1+num_frames_per_video)=column_video_segment;
        position_to_insert=position_to_insert+num_frames_per_video;
    end
    cd ..;
end



end


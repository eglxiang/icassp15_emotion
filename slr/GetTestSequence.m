function [test_sequence]=GetTestSequence(video_directory,class_num,test_index,video_mode,new_height,new_width,num_frames_per_video,do_normalize)

cd(video_directory);
exterior_folders=ls;
current_exterior_folder=exterior_folders(2+class_num,:);
cd(current_exterior_folder);
interior_files=ls;
[num_interior_files,~]=size(interior_files);
num_videos_current_emotion=(num_interior_files-2)/2;
video_name =interior_files(2+num_videos_current_emotion+test_index,:);   
load(video_name);
[test_sequence]= ProcessVideo(video_matrix,new_height,new_width,video_mode,num_frames_per_video,do_normalize); %Images are now columns of a matrix



function [num_videos_per_emotion]=CountVideosPerEmotion(video_directory)

cd(video_directory);
exterior_folders=ls;
[num_exterior_folders,~]=size(exterior_folders);
num_videos_per_emotion = zeros(1,num_exterior_folders-2);
for i=3:num_exterior_folders
    current_exterior_folder=exterior_folders(i,:);
    cd(current_exterior_folder);
    interior_files=ls;
    [num_interior_files,~]=size(interior_files);
    num_videos_per_emotion(i-2)=(num_interior_files-2)/2; % half png, half mat
    cd ..;
end


end


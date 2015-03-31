function [column_video_segment]= ProcessVideo(videomatrix,new_height,new_width,video_mode,num_output_frames,do_normalize)

videomatrix = ResizeImages2(videomatrix,new_height,new_width);
[num_frames,height,width]=size(videomatrix);
vec_video=reshape(videomatrix,[num_frames,height*width])'; % Each face is a column

%% Video Subsequences For Training 

if(strcmp(video_mode,'lasts_minus_first'))
  column_video_segment = vec_video(:,num_frames-num_output_frames+1:num_frames);
  column_video_segment = column_video_segment -vec_video(:,1)*ones(1,num_output_frames);% subtract neutral
end

if(strcmp(video_mode,'equally_spaced_minus_first'))
  indices=floor((1:1:num_output_frames)*(num_frames/num_output_frames));
  column_video_segment = vec_video(:,indices);
  column_video_segment = column_video_segment -vec_video(:,1)*ones(1,num_output_frames);
end

%% Video Subsequences For Testing 

if(strcmp(video_mode,'lasts_and_first'))
  column_video_segment = [vec_video(:,1),vec_video(:,num_frames-num_output_frames+2:num_frames)];
end

if(strcmp(video_mode,'odd_frames'))
  last_frame=2*floor((num_frames-1)/2)+1;
  column_video_segment = vec_video(:,1:2:last_frame);
end

if(strcmp(video_mode,'equally_spaced'))
  indices=floor((0:1:(num_output_frames-1))*((num_frames-1)/(num_output_frames-1)))+ones(1,num_output_frames);
  column_video_segment = vec_video(:,indices);
end

if(do_normalize==1)
    inv_norms=1./(sqrt(sum(column_video_segment.*column_video_segment,1))+0.000001); % l2 norm along column
    column_video_segment=(column_video_segment.*(ones(height*width,1)*inv_norms));
end

end


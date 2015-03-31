function [class_indices] = FindClassesIndices( indices )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
num_categories=max(indices);
inidces_length=length(indices);
class_indices=zeros(1,num_categories+1);
class_indices(1)=1;
label_pos=1;
current_val=indices(1);
for i=2:inidces_length
    if(current_val~=indices(i))
        label_pos=label_pos+1;
        class_indices(label_pos)=i;
        current_val=indices(i);
    end
end
class_indices(label_pos+1)=inidces_length+1;

end


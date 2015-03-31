clear all;
close all;

addpath(cd());
%%%%%%%%%%%%%%%%%%%%%%% TRAINING PHASE
fprintf('Generating Train and Test Indices \n');
%% Count Videos Per Emotions 
initial_directory= cd();

 %video_directory='..\DATA\SimpleCropedData\VideoGreaterThan4';
 video_directory='..\DATA\SimpleCropedData\VideoGreaterThan8';
 new_height = 64;
 new_width =  64;

%video_directory='..\DATA\SimpleCropedData\VideoGreaterThan4';
%new_height = 96/2;
%new_width = 84/2;

num_videos_per_emotion=CountVideosPerEmotion(video_directory);
cd(initial_directory);
num_classes=length(num_videos_per_emotion);

%% Set Training and Testing Indices 
train_samples_per_class = 10;
test_samples_per_class = 5;
train_indices = zeros(num_classes, train_samples_per_class);
test_indices = zeros(num_classes, test_samples_per_class);

for i=1:num_classes
     join_indices = randperm(num_videos_per_emotion(i),train_samples_per_class+test_samples_per_class); % random sample trn/tst data witin class
     train_indices(i,:) = join_indices(1:train_samples_per_class);
     test_indices(i,:) = join_indices(train_samples_per_class+1:train_samples_per_class+test_samples_per_class);  
end

%% Generate dictionary
%dictionary_mode = 'lasts_minus_first';
dictionary_mode = 'equally_spaced_minus_first';
num_frames_per_train_video = 8;
do_normalize_train=1;
fprintf('Generating Dictionary \n');
dictionary = GenerateDictionary(video_directory,train_indices,dictionary_mode,new_height,new_width,num_frames_per_train_video,do_normalize_train);
cd(initial_directory);
% visualize the dictionary
h1 = figure();
for i=1:num_classes*train_samples_per_class
    subplot(num_classes,train_samples_per_class,i);
    imshow(reshape(dictionary(:,(i-1)*num_frames_per_train_video+1),[new_height,new_width]), [ ]);
end
saveas(gcf,'..\figure\dictionary.jpg');

%%%%%%%%%%%%%%%%%%%%%%% TESTING PHASE
%test_mode = 'odd_frames';
%test_mode = 'lasts_and_first';
test_mode = 'equally_spaced';
num_frames_per_test_video = 8;
do_normalize_test=0; % why not normalized

global_max_iter=100;
lasso_max_iter=100;
alpha =10;
confussion_matrix=zeros(7,7);
num_correct_classified=0;
num_experiments_run=0;
for i=1:num_classes
    for j=1:test_samples_per_class
    fprintf('Test Task %d of %d \n',num_experiments_run,num_classes*test_samples_per_class);
    class_num=i;
    test_index=test_indices(i,j);
    test_sequence=GetTestSequence(video_directory,class_num,test_index,test_mode,new_height,new_width,num_frames_per_test_video,do_normalize_test);
    cd(initial_directory);    
    [matched_label,X_recovered,L_recovered] = ccSolveModel(i,j,num_classes,train_samples_per_class,dictionary,test_sequence,num_frames_per_train_video,global_max_iter,lasso_max_iter,alpha,new_height,new_width);
    fprintf('Label: Matched %d - Real %d \n',matched_label,i);
    if(matched_label==i)
        num_correct_classified=num_correct_classified+1;
    end
    num_experiments_run=num_experiments_run+1;
    fprintf('Partial Recognition Rate = %f \n',num_correct_classified/num_experiments_run);
    confussion_matrix(i,matched_label)=confussion_matrix(i,matched_label)+1;
    end
end

fprintf('Recognition Rate = %f \n',num_correct_classified/num_experiments_run);
fprintf('Confusion Matrix  \n');
confussion_matrix
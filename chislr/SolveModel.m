function [nearest_class_index,X_recovered,L_recovered] = SolveModel(num_classes,train_samples_per_class,train_faces,test_sequence,num_emotions_per_neutral,global_max_iter,lasso_max_iter,alpha,height,width)

% [X_recovered,L_recovered] = FastSolver(test_sequence,train_faces,alpha,global_max_iter,lasso_max_iter);
[X_recovered,L_recovered] = FastSolver3(test_sequence,train_faces,alpha,global_max_iter);
%[X_recovered,L_recovered] = Solver(test_sequence,train_faces,alpha,global_max_iter,'LassoShooting',lasso_max_iter);

% plot the recovery results
h2 = figure();
%title('1st row: testing; 2nd row: neutral face; 3rd row: expression; 4th row: recovered X');
for i=1:num_emotions_per_neutral
    subplot(5,num_emotions_per_neutral,i), imshow(reshape(test_sequence(:,i),[height width]), [ ]);
    subplot(5,num_emotions_per_neutral,(i+num_emotions_per_neutral)), imshow(reshape(L_recovered(:,i),[height width]), [ ]);
    subplot(5,num_emotions_per_neutral,(i+2*num_emotions_per_neutral)), imshow(reshape(train_faces*X_recovered(:,i),[height width]), [ ]);
    subplot(5,num_emotions_per_neutral,(i+3*num_emotions_per_neutral)), imshow(reshape(test_sequence(:,i)-L_recovered(:,i)-train_faces*X_recovered(:,i),[height width]), [ ]);
end
subplot(5,8,33:40), imagesc(X_recovered);
%saveas(gcf,strcat('..\figure\recovery\tst_', i, '.jpg'));
close(h2);

% assign label
nearest_class_distance = Inf;
nearest_class_index = -1;

for j=1:num_classes
    class_matrix=train_faces(:,num_emotions_per_neutral*train_samples_per_class*(j-1)+1:num_emotions_per_neutral*train_samples_per_class*j);
    class_representant=class_matrix*X_recovered(num_emotions_per_neutral*train_samples_per_class*(j-1)+1:num_emotions_per_neutral*train_samples_per_class*j,:);
    class_representant=class_representant+L_recovered;    
    error=norm(test_sequence-class_representant,'fro'); % residue
    fprintf('Error to class %d = %f \n',j,error);
    if(error<nearest_class_distance) % bubble rank
        nearest_class_distance=error;
        nearest_class_index = j;
    end
end
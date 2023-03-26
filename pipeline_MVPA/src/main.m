function main()
fprintf('INITIALIZING')

addpath('/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/matlab_svc/')
savepath
X = load('X_data.mat')
Y = load('Y_data.mat')
id = load('gr_data.mat')
disp(X)
disp(Y.Y)
disp(id.gr)


addpath('/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/CanlabCore/Statistics_tools/Cross_validated_Regression/')

S = xval_SVM(X.X, cell2mat(Y.Y)',id.gr, 'noplot','dorepeats',5);
print('done with function')

disp(S)
addpath('/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/')
save('struct.mat', '-struct', 'S')
print('SUCKKKKKKKKKKKKKKKKAAAAAAAAAAAAAAAAA WHOUWHOUUUUUUUUUUUU')
end

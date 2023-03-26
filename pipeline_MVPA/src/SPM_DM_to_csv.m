
MainDirectory = 'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\TxT_SPM_MAT_Files'; % define a main directory

ResultFiles = dir(fullfile(MainDirectory,'**\APM*')); % recursivly seach through subfolders looking for results.txt files

outDir = 'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_single_event_csv'

for i = 1:24 %24 is the length of the files, this should be changed to be more general

CurrentFile = fullfile(ResultFiles(i).folder,ResultFiles(i).name);
%disp(CurrentFile)
target_file = fullfile(CurrentFile,'SPM.mat')
SPM = load(target_file)
design_matrix = SPM.SPM.xX.X;
cols = SPM.SPM.xX.name;
disp(length(design_matrix))
%save
DM_name = 'DM.csv'
col_names = 'events.csv'

save_dir = fullfile(outDir,ResultFiles(i).name,DM_name)
writematrix(design_matrix, save_dir)

save_dir = fullfile(outDir,ResultFiles(i).name,col_names)
writecell(cols, save_dir)

end
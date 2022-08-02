import os
#path = os.getcwd()
path = r'/data/rainville/dylan_projet_ivado_decodage/results/GLM_1st_level_each_shock'

#print(path)
#path = r'C:\Users\Dylan\Desktop\UdeM_A21'

#def count_file_in_dir(path):
    
for file in os.listdir(path):
    count = 0
    print(file)
    for items in os.listdir(os.path.join(path,file)):
        count += 1
    print(count)

#count_file_in_dir(path)

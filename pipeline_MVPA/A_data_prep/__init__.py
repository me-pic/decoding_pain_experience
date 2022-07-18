class functions_data:


    def get_subj_data(data_dir, prefix):

        #//////////description/////////////
        #fonction pour filtrer les volumes selon un préfix parmi plusieurs volumes dans un dossier

        #data_dir: le path vers tous les volumes pour un sujet
        #prefix : le préfix avec lequel on veut filtrer les fichiers

        #si on met 'sw', om va avoir tous les volumes avec ce préfixe et la focntion retourne les données pour ce sujet

        #Fonction qui retourne une liste avec le path de toutes les nii dans un dossier
        #avec un préfix

        #//////////main/////////////

        import os
        #Extraction des volumes pour un sujet dans une liste
        ls_volumes_all = os.listdir(data_dir)


        #Crée une liste avec seulement les fichiers commencant par sw
        swaf_list = [x for x in ls_volumes_all if x.startswith(prefix)]
        #print(swaf_list)

        #joindre le path avec les noms des volumes dans une liste
        #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
        subject_data = [os.path.join(data_dir, name) for name in swaf_list]


        print('len(subject_data): {} '.format(len(subject_data)))

        return subject_data


    #=================================================================
    def split_reg_hypo(matrix_to_split, target_lenght):

        #funciton qui split une matrice (matrix_to_split)selon le nombre de volumes qu'on donne en argument (target_lenght)

         #on split horizontalement la matrice en une tranche de
         split_matrix = matrix_to_split.iloc[0:target_lenght, :]

         return split_matrix


    #=================================================================
    def split_reg_hyper(matrix_to_split, target_lenght):

        #funciton qui split une matrice (matrix_to_split)selon le nombre de volumes qu'on donne en argument (target_lenght)

        inverse_lenght = len(matrix_to_split) - target_lenght
        #on split horizontalement la matrice en une tranche de
        split_matrix = matrix_to_split.iloc[inverse_lenght :len(matrix_to_split), :]

        return split_matrix

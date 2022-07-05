def get_subj_data(data_dir, prefix):

    #//////////description/////////////

    #Fonction qui retourne une liste avec le path de toutes les nii dans un dossier
    #avec un préfix spécifique

    #data_dir: le path vers tous les volumes pour un sujet
    #prefix : le préfix avec lequel on veut filtrer les fichiers

    #si on met 'sw', om va avoir tous les volumes avec ce préfixe et la fonction retourne les données pour ce sujet


    #///////////////////////

    import os

    #Extraction des volumes pour un sujet dans une liste
    ls_volumes_all = os.listdir(data_dir)
    
    #Crée une liste avec seulement les fichiers commencant par sw
    swaf_list = [x for x in ls_volumes_all if x.startswith(prefix)]
    #print(swaf_list)

    #joindre le path avec les noms des volumes dans une liste
    #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
    subject_data = [os.path.join(data_dir, name) for name in swaf_list]



    return subject_data


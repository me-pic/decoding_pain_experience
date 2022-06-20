def split_reg_hypo(matrix_to_split, target_lenght):

    #funciton qui split une matrice (matrix_to_split)selon le nombre de volumes qu'on donne en argument (target_lenght)
    
     #on split horizontalement la matrice en une tranche de 
     split_matrix = matrix_to_split.iloc[0:target_lenght, :]

     return split_matrix


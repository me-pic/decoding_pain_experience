def split_reg_hyper(matrix_to_split, target_lenght):

    #funciton qui split une matrice (matrix_to_split)selon le nombre de volumes qu'on donne en argument (target_lenght)
    
    inverse_lenght = len(matrix_to_split) - target_lenght
    #on split horizontalement la matrice en une tranche de
    split_matrix = matrix_to_split.iloc[inverse_lenght :len(matrix_to_split), :]

    return split_matrix

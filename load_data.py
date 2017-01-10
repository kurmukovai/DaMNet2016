import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix


def convert(data, size=68, mode = 'vec2mat'): #diag=0,
    '''
    Convert data from upper triangle vector to square matrix or vice versa
    depending on mode.
    INPUT : 
    data - vector or square matrix depending on mode
    size - preffered square matrix size (given by formula :
           (1+sqrt(1+8k)/2, where k = len(data), when data is vector)
    diag - how to fill diagonal for vec2mat mode
    mode - possible values 'vec2mat', 'mat2vec'
    OUTPUT : 
    square matrix or 1D vector 
    EXAMPLE :
    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    vec_a = convert(a, mode='mat2vec')
    print(vec_a)
    >>> array([2, 3, 4])
    convert(vec_a, size = 3, diag = 1, mode = vec2mat)
    >>> matrix([[1, 2, 3],
                [2, 1, 4],
                [3, 4, 1]], dtype=int64)
    '''

    if mode == 'mat2vec':
        
        mat = data.copy()
        rows, cols = np.triu_indices(data.shape[0],k = 0)
        vec = mat[rows,cols]
        
        return vec

    elif mode == 'vec2mat':
        
        vec = data.copy()        
        rows, cols = np.triu_indices(size,k = 0)
        mat = csr_matrix((vec, (rows, cols)), shape=(size, size)).todense()
        mat = mat + mat.T # symmetric matrix
        np.fill_diagonal(mat, np.diag(mat)/2)
        
    return mat

def load_adni(path):
    '''
    Simple script to import ADNI data set
    
    Данный набор данных содержит 807 снимков для 255 пациентов,
    каждому снимку поставлен в соответствие граф размера с 68 вершинами,
    метка класса (EMCI, Normal, AD, LMCI, SMC), а так же метка пациентов 
    (так как для каждого пациента есть несколько снимков,
    метки класса для одного пациента одинаковы для всех его снимков)
    
    IMPUT :
    
    path - this folder should contain 2 folders ("matrices" and "adni2_centers")
           and 1 excel file ("ADNI2_Master_Subject_List.xls")
           
    OUTPUT : 
    
    data - numpy array of shape #subjects x #nodes x #nodes
    target - numpy array containing target variable
    data_copy - pandas dataframe containing 
                subject_id, 
                scan_id (multiple scans for some patients),
                adjacency matrices (data converted to vectors)
                target (diagnosis - AD, Normal, EMCI, LMCI, SMC)
                
    EXAMPLE : 
    
    path = 'notebooks to sort/connectomics/ADNI/Data'
    data, target, info = load_adni(path)
    
    
    TODO : 
    
    Add physical nodes position
    '''
    
    
    path_matrices = path + '/matrices/'
    path_subject_id = path + '/ADNI2_Master_Subject_List.xls'
    
    all_matrices = pd.DataFrame(columns = ['subject_id_file','subject_id','scan_id', 'matrix', 'target'])

    # import data
    for foldername in sorted(os.listdir(path_matrices)):
        for filename in sorted(os.listdir(path_matrices+foldername)):
            if 'NORM' not in filename:
                mat = np.genfromtxt(path_matrices+foldername+'/'+filename)
                subject_id_file = foldername
                subject_id = foldername[:-2]
                scan_id = foldername[-1:]

                # ADNI data have zeros on 3 and 38 row and column
                mat = np.delete(mat, [3,38], 1) 
                mat = np.delete(mat, [3,38], 0)

                subject_data = convert(mat, mode = 'mat2vec')
                single_subject = pd.DataFrame(data = [[subject_id_file, subject_id, scan_id, subject_data, np.nan]],
                                              columns = ['subject_id_file','subject_id','scan_id', 'matrix', 'target'])
                all_matrices = all_matrices.append(single_subject)

    all_matrices.index = all_matrices.subject_id_file
    subject_data = pd.read_excel(path_subject_id, sheetname = 'Subject List')
    subject_id_names = np.array(all_matrices['subject_id_file'])

    #importing target variables
    for name in subject_id_names:
        smth = subject_data.loc[subject_data['Subject ID'] == name[:-2]]['DX Group'].dropna()
        un_smth = np.unique(smth)
        try:
            val = un_smth[0].replace(' ', '')
            all_matrices.set_value(name, 'target', val)
        except:
            pass

    #drop objects without any target
    all_matrices.dropna(inplace = True)
    data_copy = all_matrices.copy()



    temp = data_copy['matrix']

    data_vectors = np.zeros((807, 2346))
    data = np.zeros((807, 68, 68))

    for idx, vec in enumerate(temp):
        data_vectors[idx] = vec
        data[idx] = convert(vec)

    target = all_matrices.target.values
    patients_ids = data_copy.subject_id.values

    print('ADNI data shape                   :', data.shape,
         '\nADNI target variable shape        :', target.shape,
         '\nADNI number of unique patients    :', data_copy.subject_id.unique().shape)
    return data, target, data_copy



def load_ucla(path):
    '''
    Simple script to import UCLA Autism data set
    
    Набор данных UCLA содержит 94 снимка (людей с аутизмом и без),
    по 1 снимку для каждого пациента, каждому снимку поставлен в соответствие граф
    с 264 вершинами, переменная target содержит метки классов (1 - ASD - Аутизм, 0 - TD - норма)
    
    
    IMPUT :
    
    path - this folder should contain 2 folders ("dti" and "func")
           
    OUTPUT : 
    
    data - numpy array of shape #subjects x #nodes x #nodes
    target - numpy array containing target variable (1 stands for AutismSpectrumDisorder, 0 for TypicalDevelopment)
    
    EXAMPLE : 
    
    path = 'notebooks to sort/connectomics/Autism/Data/
    data, target = load_ucla(path)
    
    
    TODO : 
    
    Add physical nodes position
    '''
    path = path + 'dti/'
    target_vector = [] #this will be a target vector (diagnosis)
    matrices=[] # this will be a list of connectomes 

    for filename in sorted(os.listdir(path)): #for each file in a sorted (!) list of files:
        if "DTI_connectivity" in filename: #we only need files with DTI connectivity matrices
            if "All" not in filename: #we also do not need an average connectivity matrix here
                A_dataframe = pd.read_csv(path + filename, sep = '   ', header = None, engine = 'python')
                A = A_dataframe.values # we will use a list of numpy arrays, NOT pandas dataframes
                matrices.append(A) #append a matrix to our list
                if "ASD" in filename:
                    target_vector.append(1)
                elif "TD" in filename: 
                    target_vector.append(0)
    data = np.array(matrices)
    target = np.array(target_vector)

    print('UCLA Autism data shape                   :', data.shape,
         '\nUCLA Autism target variable shape        :', target.shape)

    return data, target

def load_apoe(path):
    '''
    Simple script to import UCLA APOE data set
    
    Набор данных UCLA APOE содержит 55 снимков 
    (людей носителей алелли а4, наличие которой повышает
    вероятность возникновения болезни Паркинсона, и без нее)
    по 1 снимку для каждого человека, каждому снимку поставлен
    в соответствие граф со 110 вершинами, переменная target
    содержит метки классов (1 - носитель, 0 - нет)
    
    IMPUT :
    
    path - this folder should contain 55 .txt files with adjecency matrices
           
    OUTPUT : 
    
    data - numpy array of shape #subjects x #nodes x #nodes
    target - numpy array containing target variable (1 stands for 
             one who has apoe4 , 0 for one who has not)
    xyz - numpy array of shape #nodes x 3 - nodes euclidean coordinates
    
    EXAMPLE : 
    
    path = 'notebooks to sort/UCLA_APOE_matrices/
    data, target = load_apoe(path)
    
    TODO : 
    
    Add physical nodes position
    '''
    #path = 'notebooks to sort/UCLA_APOE_matrices/' #put your correct path here
    target_vector_a4 = [] #this will be a target vector (diagnosis)
    matrices_a4=[] # this will be a list of connectomes 

    for filename in sorted(os.listdir(path)): #for each file in a sorted (!) list of files:
        if "connectivity" in filename: #we only need files with DTI connectivity matrices
            if "All" not in filename: #we also do not need an average connectivity matrix here
                A_dataframe = pd.read_csv(path + '/' + filename, sep = '   ', header = None, engine = 'python')
                A = A_dataframe.values # we will use a list of numpy arrays, NOT pandas dataframes
                matrices_a4.append(A) #append a matrix to our list
                if "APOE-4" in filename:
                    target_vector_a4.append(1)
                elif "APOE-3" in filename: 
                    target_vector_a4.append(0)

    data = np.array(matrices_a4)
    target = np.array(target_vector_a4)
    
    xyz = np.loadtxt(path + 'APOE-3_1_region_xyz_centers_file.txt')

    print('UCLA APOE data shape                   :', data.shape,
         '\nUCLA APOE target variable shape        :', target.shape)
    return data, target, xyz


def load_parkinson(path):
    
    '''
    Simple script to import parkinson data set
    
    Данный набор данных содержит 553 снимка для 295 пациентов,
    каждому снимку поставлен в соответствие граф размера с 68 вершинами,
    метка класса (Prodromal, SWEDD, PD, Control, GenCohortUnaff, GenCohortPD),
    а так же метка пациентов (так как для каждого пациента есть несколько снимков,
    метки класса для одного пациента одинаковы для всех его снимков <-- не проверено)
    
    IMPUT :
    
    path - this folder should contain folder Data, and file demo_info.txt
           
    OUTPUT : 
    
    data - numpy array of shape #subjects x #nodes x #nodes
    target - numpy array containing target variable 
    all_matrices - pandas dataframe containing physical centers of nodes
    
    EXAMPLE : 
    
    path = 'notebooks to sort/connectomics/Parkinson/'
    data, target = load_apoe(path)

    '''
    #path = 'notebooks to sort/connectomics/Parkinson/Data/'
    
    meta_data = pd.read_csv(path + 'demo_info.txt',header=None)
    meta_data.columns = ['subject_id_file', 'id', 'date','age','sex', 'target']
    meta_data.index = meta_data.subject_id_file
    
    path = path + 'Data/'
    
    all_matrices = pd.DataFrame(columns = ['subject_id_file','subject_id',
                                           'matrix','centers', 'target'])
    data = []

    for foldername in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(path+foldername)):
            if 'FULL' in filename:
                mat = np.genfromtxt(path+foldername+'/'+filename)
                subject_id_file = foldername
                subject_id = subject_id_file[:8]
                mat = mat[:70][:,:70]
                mat = np.delete(mat, [3,38], 1)
                mat = np.delete(mat, [3,38], 0)
                data.append(mat)
                subject_data = convert(mat,size = 68, mode = 'mat2vec')

            elif 'connect_grav' in filename:
                centers = pd.read_csv(path+foldername+'/'+filename)
                centers.drop([3,38], inplace=True)
                subject_center = np.array(centers[['mm_cordX', 'mm_cordY', 'mm_cordZ']])


        single_subject = pd.DataFrame(data = [[subject_id_file, subject_id, subject_data, subject_center, np.nan]],
                                      columns = ['subject_id_file','subject_id', 'matrix','centers', 'target'])
        all_matrices = all_matrices.append(single_subject)
    all_matrices.index = all_matrices.subject_id_file


    

    all_matrices.target = meta_data.target

    data = np.array(data)
    target = all_matrices.target.values
    patients_ids = all_matrices.subject_id.values

    print('Parkinson  data shape                  :', data.shape,
         '\nParkinson target variable shape        :', target.shape,
         '\nParkinson number of unique patients    :', all_matrices.subject_id.unique().shape)
    
    return data, target, all_matrices

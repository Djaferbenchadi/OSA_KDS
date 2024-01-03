#  %%
import subprocess
import hdf5storage
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

start_time = time.time()

def run_script(sgm, subdim, gdsdim):
  cmd = ['python', 'script.py', '--sgm', str(sgm), '--subdim', str(subdim), '--gdsdim', str(gdsdim)]
  subprocess.run(cmd)

# List of parameter sets for Hyperparameter optimization (alpha, subspace_dimension, DS_dimension) 
parameter_sets = [(0.1, i, 163 + 11*(i - 15)) for i in range(15, 61)]


train_x = []
test_x_ = []

for i in range(1, 12):
    file_name = "X" + str(i) + ".mat"
    file_path = "Dumpware_multiple/" + file_name
    data = hdf5storage.loadmat(file_path)
    variable_name = "X"+str(i)
    data_transposed = np.array(data[variable_name].T)
    train_x.append(data_transposed)

for i in range(1, 12):
    file_name = "Y" + str(i) + ".mat"
    file_path = "Dumpware_multiple/" + file_name
    data = hdf5storage.loadmat(file_path)
    variable_name = "Y"+str(i)
    data_transposed = np.array(data[variable_name].T)
    test_x_.append(data_transposed)
    
train_y = np.arange(11)


test_x_all = np.concatenate((test_x_), axis = 0)
test_x = [test_x_all[i, np.newaxis, :] for i in range(test_x_all.shape[0])]


test_x = test_x[5:6]
test_y = [[0]]

Accuracy = []
dimension_safe = []
dimension_mal = []

####################################################################
image_shape = (32, 32)
image = test_x[0].reshape(image_shape)
heatmap_colors = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_HOT)
heatmap = np.zeros(image.shape[:2], dtype=np.float32)

step_size = 1
window_size = 1

# Compute occlusion sensitivity scores for each pixel

for l in range(0, image_shape[0]-window_size+1, int(step_size)):
    for j in range(0, image_shape[1]-window_size+1, int(step_size)):
        
        image_ = image.T
        # Create occluded image with pixel (i,j) set to zero
        occluded_image = np.copy(image_)
        occluded_image[l:l+window_size, j:j+window_size] = 0
        # plt.imshow(occluded_image, cmap='gray')
        # plt.title('Occluded image')
        # plt.show()
        ####if we work with (224,224) ocludded image
        # occluded_image_ = occluded_image.reshape(1, 50176)
        occluded_image_ = occluded_image.T
        # plt.imshow(occluded_image_, cmap='gray')
        # plt.title('Occluded image')
        # plt.show()
        occluded_image_ = occluded_image_.reshape(1, 1024)
        test_x_oc = [occluded_image_]
        test_y = [[0]]
        ####################################################################


        for sgm, subdim, gdsdim in parameter_sets:
            run_script(sgm, subdim, gdsdim)
            for var in list(globals().keys()):
              if var not in ['function variables','accuracy_score','gauss_class_mat_diff','gauss_gram_mat','gauss_gram_two','gauss_projection_diff','get_ipython','jit','parentpath1','rand','randint','run_script','trace_this_thread','train_test_split','special variables','__','___','__doc__','__file__','__loader__','__name__','__package__','__spec__','__vsc_ipynb_file__','__builtin__','__builtins__','debugpy','exit','hdf5storage','np','os','pd','quit','random','subprocess','sys','_VSCODE_hashlib','_VSCODE_types','_dh','_VSCODE_compute_hash','_VSCODE_wrapped_run_cell','normalize','sgm', 'subdim', 'train_x','train_y','test_x','test_x_oc','test_y','gdsdim', 'time', 'start_time','image_shape','image','image_','parameter_sets','l','heatmap','heatmap_colors','i_start','i_end', 'j_start','j_end','win_size','i','j','step_size','window_size']:
                del globals()[var]

            import os
            import numpy as np
            from numpy.random import randint, rand
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            import hdf5storage
            import random
            import pandas as pd
            from numpy.random import randint, rand
            from numba import jit, void, f8, njit
            from sklearn.preprocessing import normalize
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support


            count1 = 0
            trainBasis = []
            for train_x_i in train_x:
                train_self = train_x_i.T @ train_x_i
                w, v = np.linalg.eigh(train_self)
                w, v = w[::-1], v[:, ::-1]
                rank = np.linalg.matrix_rank(train_self)
                w, v = w[:rank], v[:, :rank]
                base = v[:, 0:subdim]
                trainBasis.append(base)

            # Ds + projection 
            allbase = np.zeros((train_x[0].shape[1], train_x[0].shape[1]))
            for subspace_i in trainBasis:
                allbase += subspace_i @ subspace_i.T

            w, v = np.linalg.eigh(allbase)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(allbase)
            w, v = w[:rank], v[:, :rank]
            gds = v[:, v.shape[1]-gdsdim:v.shape[1]]

            subspace_ongds_list = []
            for subspace_i in trainBasis:
                bases_proj = np.matmul(gds.T, subspace_i)
                qr = np.vectorize(np.linalg.qr, signature='(n,m)->(n,m),(m,m)')
                bases, _ = qr(bases_proj)
                subspace_ongds_list.append(bases)
            #------------------------------------------------------------------------------------------


            #calculate projection length--------------------------------------------------------------
            similarity_all = []
            similarity_original = []
            similarity_OSA = []
            
            test_x_ii = test_x[0]
            test_x_ongdss = np.matmul(gds.T, test_x_ii.T)
            length = np.linalg.norm(test_x_ongdss)
            similarity_original.append(length)  
            print("similarity_original:")    
            print(similarity_original)
            ##################################################


            for test_x_i in test_x_oc:
                test_x_ongds = np.matmul(gds.T, test_x_i.T)
                # Mesure the length of the projected input into DS
                length_OSA = np.linalg.norm(test_x_ongds)
                similarity_OSA.append(length_OSA)
                print("similarity_OSA:")    
                print(similarity_OSA)
              ##################################################
            heatmap[l:l+window_size, j:j+window_size] += similarity_OSA[0]
            #------------------------------------------------------------------------------------------


heatmap /= np.max(heatmap)
plt.imshow(heatmap, cmap='RdBu')
plt.colorbar()
plt.show()


# %%

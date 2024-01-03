#  %%
import subprocess
import hdf5storage
import numpy as np
import time
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2

start_time = time.time()

def run_script(sgm, subdim, gdsdim):
  cmd = ['python', 'script.py', '--sgm', str(sgm), '--subdim', str(subdim), '--gdsdim', str(gdsdim)]
  subprocess.run(cmd)

# List of parameter sets for Hyperparameter optimization (alpha, subspace_dimension, DS_dimension) 
parameter_sets = [(0.1, 6, 10)]


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
    
X_safe = hdf5storage.loadmat('X_safe.mat')
X_safe = np.array(X_safe['X_safe']).T
X_safe = X_safe[:300]

train_x = train_x[1]
train_x = [train_x, X_safe] 

train_y = np.array([1, 2])


test_x_all = np.concatenate((test_x_), axis = 0)
test_x = [test_x_all[i, np.newaxis, :] for i in range(test_x_all.shape[0])]


test_x = test_x[94:95] ## Select single malware input from dataset
test_y = [[0]]
Accuracy = []
dimension_safe = []
dimension_mal = []

####################################################################
image_shape = (32, 32)
image = test_x[0].reshape(image_shape)
heatmap_colors = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_HOT)
heatmap = np.zeros(image.shape[:2], dtype=np.float32)

step_size = (1, 1)
window_size = (1, 1)


# Compute occlusion sensitivity scores for each pixel

for l in range(0, image_shape[0] - window_size[0] + 1, int(step_size[0])):
    for j in range(0, image_shape[1] - window_size[1] + 1, int(step_size[1])):

        image_ = image.T
        occluded_image = np.copy(image_)
        occluded_image[l:l+window_size[0], j:j+window_size[1]] = 0
        
        # occluded_image[l:l+window_size, j:j+window_size] = 0
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
            from numba import jit, void, f8, njit

            @jit(void(f8[:, :], f8[:, :]))
            def gauss_gram_mat(x, K):
              n_points = len(x)
              n_dim = len(x[0])
              b = 0

              for j in range(n_points):
                for i in range(n_points):
                  for k in range(n_dim):
                    b = (x[i][k] - x[j][k])
                    K[i][j] += b * b

            @jit
            def gauss_class_mat_diff(K_d):
                for c1 in range(class_num):
                    for c2 in range(class_num):
                        for i1 in range(subdim):
                            for j2 in range(subdim):
                                for s1 in range(class_info[c1]):
                                    for t2 in range(class_info[c2]):
                                        K_d[subdim * c1 + i1, subdim * c2 + j2] += K_cl[class_index[c1] + s1, i1] * K_cl[class_index[c2] + t2, j2] * K_all[class_index[c1] + s1, class_index[c2] + t2]

            @jit
            def gauss_projection_diff(x, K_p):
                for i_data in range(x.shape[0]):
                    for i_gds in range(b.shape[1]):
                        for c1 in range(class_num):
                            for i1 in range(subdim):
                                for s1 in range(class_info[c1]):
                                    K_p[i_gds, i_data] += b[i_gds, subdim * c1 + i1] * K_cl[class_index[c1] + s1, i1] * x[i_data, class_index[c1] + s1]

            @jit
            def gauss_gram_two(X, Y, K_t):
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        for k in range(X.shape[1]):
                            b = (X[i][k] - Y[j][k])
                            K_t[i][j] += b * b


            def dual_vectors(K, n_subdims=None, higher=True, elim=False, eps=1e-6):
                e, A = np.linalg.eigh(K)
                e[(e < eps)] = eps

                A = A / np.sqrt(e)

                if elim:
                    e = e[(e > eps)]
                    A = A[:, (e > eps)]

                if higher:
                    return A[:, -n_subdims:]

                return A[:, :n_subdims]


            print("sigma:",sgm,"subdim:",subdim,"gdsdim:",gdsdim)   

            #parameter-------------------------------------------------------------------------------
            class_num = 2                            
            class_info = np.array([364, 300])   
            #----------------------------------------------------------------------------------------


            #normalization---------------------------------------------------------------------------
            class_index = []
            count_c = 0
            for class_i in class_info:
                if count_c == 0:
                    class_index.append(0)
                    class_index.append(class_info[0])
                else:
                    class_index.append(class_index[count_c] + class_info[count_c])
                count_c += 1
            class_index = np.array(class_index)


            K_class = []
            for train_data in train_x:
                K = np.zeros((train_data.shape[0], train_data.shape[0]))
                gauss_gram_mat(train_data, K)
                K = np.exp(- K / (2 * sgm))
                X1_coeffs = dual_vectors(K, n_subdims=subdim)
                K_class.append(X1_coeffs)


            #kernel for generalized difference subspaces----------------------------------------------
            count_k = 0
            for i in range(class_num):
                if i == 0:
                    K_cl = K_class[i]
                else:
                    K_cl = np.concatenate([K_cl, K_class[i]])     

            K_D = np.zeros((subdim * class_num, subdim * class_num)) 

            for i in range(class_num):
                if i == 0:
                    train_all = train_x[i]
                else:
                    train_all = np.concatenate([train_all, train_x[i]]) 

            K_all = np.zeros((train_all.shape[0], train_all.shape[0]))                           
            gauss_gram_mat(train_all, K_all) 
            K_all = np.exp(- K_all / (2 * sgm)) 

            gauss_class_mat_diff(K_D) 
            B, b = np.linalg.eigh(K_D)
            b = b[:, 0:gdsdim]
            print("training complete")

            #prepare test data, project data onto KDS--------------------------------------------------------------------
            test_np = np.array(test_x_oc)
            test_np = test_np.reshape([test_np.shape[0], test_np.shape[2]])

            train_data_projected = np.zeros((gdsdim, train_all.shape[0]))
            gauss_projection_diff(K_all, train_data_projected)
            print("projection complete 1 (train(class) data)")

            K_two = np.zeros((test_np.shape[0], train_all.shape[0]))
            gauss_gram_two(test_np, train_all, K_two)
            K_two = np.exp(- K_two / (2 * sgm))


            test_data_projected = np.zeros((gdsdim, test_np.shape[0]))
            gauss_projection_diff(K_two, test_data_projected)
            ##################################################


            #Generate linear subspace for train data projected onto KDS-------------------------------
            subspace_class = []
            for i in range(class_num):
                data_projected = train_data_projected[:, class_index[i]:class_index[i+1]]
                w, v = np.linalg.eigh(data_projected @ data_projected.T)
                w, v = w[::-1], v[:, ::-1]
                rank = np.linalg.matrix_rank(K)
                w, v = w[:rank], v[:, :rank]
                d = v[:, 0:subdim]
                subspace_class.append(d)
            #-----------------------------------------------------------------------------------------

            #calculate projection length--------------------------------------------------------------
            similarity_all = []
            
            for i in range(test_np.shape[0]):
                data_input = test_data_projected[:, i]
                similarity_one = []
                
                length = np.linalg.norm(subspace_class[0].T @ data_input, ord = 2)
                similarity_one.append(length)
            similarity_all.append(similarity_one)
            print("input length:")
            print(similarity_one)
            heatmap[l:l+window_size[0], j:j+window_size[1]] += similarity_one
            ###############################normal###########################################



heatmap /= np.max(heatmap)
plt.imshow(heatmap, cmap='RdBu')
plt.colorbar()
plt.show()


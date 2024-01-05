import numpy as np
np.random.seed(1000)

from scipy.io import loadmat
daily_sport = r"../daily_sport/"

''' Load train set '''
data_dict = loadmat(daily_sport + "/daily_sport.mat")
X_train_mat = data_dict['X_train'][0]
y_train_mat = data_dict['Y_train'][0]
X_test_mat = data_dict['X_test'][0]
y_test_mat = data_dict['Y_test'][0]

y_train = y_train_mat.reshape(-1, 1)
y_test = y_test_mat.reshape(-1, 1)

var_list = []
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_list.append(var_count)

var_list = np.array(var_list)
max_nb_timesteps = var_list.max()
min_nb_timesteps = var_list.min()
median_nb_timesteps = np.median(var_list)

print('max nb timesteps train : ', max_nb_timesteps)
print('min nb timesteps train : ', min_nb_timesteps)
print('median_nb_timesteps nb timesteps train : ', median_nb_timesteps)

X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    X_train[i, :, :var_count] = X_train_mat[i]

# ''' Load test set '''

X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_test_mat.shape[0]):
    var_count = X_test_mat[i].shape[-1]
    X_test[i, :, :var_count] = X_test_mat[i][:, :max_nb_timesteps]

# ''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))




from sklearn.model_selection import train_test_split
X_test,X_val = train_test_split(X_test,test_size=0.5,random_state=42)
y_test,y_val = train_test_split(y_test,test_size=0.5,random_state=42)

print("Train dataset : ", X_val.shape, y_val.shape)
print("Test dataset : ", X_test.shape, y_test.shape)


# 如果是（K，1）维度的话，在训练时会报错
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
y_val = np.squeeze(y_val)

seq_lens_train = np.asarray([i.shape[0] for i in X_train])
seq_lens_test = np.asarray([i.shape[0] for i in X_test])
seq_lens_val = np.asarray([i.shape[0] for i in X_val])

print("seq_lens_train : ", seq_lens_train.shape)
print("seq_lens_train : ", seq_lens_train)


np.save(daily_sport + 'X_train.npy', X_train)
np.save(daily_sport + 'y_train.npy', y_train)
np.save(daily_sport + 'X_test.npy', X_test)
np.save(daily_sport + 'y_test.npy', y_test)
np.save(daily_sport + 'X_val.npy', X_val)
np.save(daily_sport + 'y_val.npy', y_val)
np.save(daily_sport + 'seq_lens_train.npy', seq_lens_train)
np.save(daily_sport + 'seq_lens_test.npy', seq_lens_test)
np.save(daily_sport + 'seq_lens_val.npy', seq_lens_val)

import numpy as np

train_val_X = np.load("dataset/train_val_X.npy")
toy_train_val_X = train_val_X[:500]
np.save("toy_dataset/train_val_X", toy_train_val_X)

train_val_y = np.load("dataset/train_val_y.npy")
toy_train_val_y = train_val_y[:500]
np.save("toy_dataset/train_val_y", toy_train_val_y)

inputs = np.load("dataset/input.npz")
keys = list(inputs.keys())
for k in keys:
    print(inputs[k].shape)

# ['train_X',
#  'valid_X',
#  'test_X',
#  'train_y',
#  'valid_y',
#  'test_y',
#  'train_error',
#  'valid_error',
#  'test_error']

np.savez("toy_dataset/input.npz",
         train_X=inputs['train_X'][:500],
         valid_X=inputs['valid_X'][:500],
         test_X=inputs['test_X'][:500],
         train_y=inputs['train_y'][:500],
         valid_y=inputs['valid_y'][:500],
         test_y=inputs['test_y'][:500],
         train_error=inputs['train_error'][:500],
         valid_error=inputs['valid_error'][:500],
         test_error=inputs['test_error'][:500])


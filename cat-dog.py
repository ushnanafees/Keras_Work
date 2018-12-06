import os, shutil

original_dataset_train_dir = 'C:/Users/Ushna/.keras/datasets/cat-dog/train'
original_dataset_test_dir = 'C:/Users/Ushna/.keras/datasets/cat-dog/test1'

# --------------- Create new folder sliced-cat-dog -------------------------
base_dir = 'E:/SSUET_WORK/sliced-cat-dog'
#os.mkdir(base_dir)

# -------- Create sub folders named as train, test and validation ---------
train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

# -------------- Creating a subfolder "cats" in train folder --------------
train_cats_dir = os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)

# -------------- Creating a subfolder "dogs" in train folder -----------------
train_dogs_dir = os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

# ------------ Creating a subfolder "cats" in validation folder --------------
validation_cats_dir = os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)

# ----------- Creating a subfolder "dogs" in validation folder ----------------
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

# ------------ Creating a subfolder "cats" in test folder ---------------------
test_cats_dir = os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)

# -------------- Creating a subfolder "dogs" in test folder -----------------
test_dogs_dir = os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)
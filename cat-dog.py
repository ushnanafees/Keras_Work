import os, shutil

original_dataset_train_dir = 'C:/Users/Ushna/.keras/datasets/cat-dog/train'
original_dataset_test_dir = 'C:/Users/Ushna/.keras/datasets/cat-dog/test1'

# --------------- Create new folder sliced-cat-dog -------------------------
base_dir = 'E:/SSUET_WORK/sliced-cat-dog'
os.mkdir(base_dir)

# -------- Create sub folders named as train, test and validation ---------
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# -------------- Creating a subfolder "cats" in train folder --------------
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# -------------- Creating a subfolder "dogs" in train folder -----------------
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# ------------ Creating a subfolder "cats" in validation folder --------------
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# ----------- Creating a subfolder "dogs" in validation folder ----------------
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# ------------ Creating a subfolder "cats" in test folder ---------------------
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# -------------- Creating a subfolder "dogs" in test folder -----------------
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)


''' --------------- Copying first 1000 images of cats 
in "sliced-cat-dog/train/cats" folder for training ----------------- ''' 

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

''' --------------- Copying 500 images after the series of 1000 images
of cats in "sliced-cat-dog/validation/cats" folder for validation ---------''' 

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)    

   
''' --------------- Copying 500 images after the series of 1500 images
of cats in "sliced-cat-dog/test/cats" folder for testing ---------'''     

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)    


''' --------------- Copying first 1000 images of dogs 
in "sliced-cat-dog/train/dogs" folder for training ----------------- ''' 

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
''' --------------- Copying 500 images after the series of 1000 images of dogs
in "sliced-cat-dog/validtion/dogs" folder for validation ----------------- ''' 
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)    


''' --------------- Copying 500 images after the series of 1500 images
of cats in "sliced-cat-dog/test/dogs" folder for testing ---------'''    

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_train_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))



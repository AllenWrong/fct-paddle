file="./data_store"

if [ ! -d $file ]; then
  echo "create $file"
  mkdir $file
  wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -P $file
  tar -xvf ./data_store/cifar-100-python.tar.gz -C ./data_store/
  python prepare_dataset.py
else
  echo "find $file, you may have cifar100!"
fi
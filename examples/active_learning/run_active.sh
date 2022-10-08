python extract_id_from_loss.py 
python write_id_lists.py 

FILES="./id_lists/*k.pth"
for f in $FILES
do
  echo "Processing $f file..."
  python train_cifar.py --config-file default_config.yaml --data.train_ids $f
done
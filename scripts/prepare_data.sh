WHOAMI=$(whoami)
DATASET_DIR=/cluster/scratch/$WHOAMI/virtual_humans_data/datasets
FACE_DIR=$DATASET_DIR/face

# mkdir $DATASET_DIR
# python3 face_data_downloader.py -d compressed -o $DATASET_DIR --not_altered --not_mask
# mv $DATASET_DIR/FaceForensics_compressed $FACE_DIR
python3 face_data_preprocessor.py --videos_dir $FACE_DIR

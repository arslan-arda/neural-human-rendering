# source $CONDA_PREFIX/etc/profile.d/conda.sh
# conda activate virtual_humans

WHOAMI=$(whoami)
DATASET_DIR=datasets
FACE_DIR=$DATASET_DIR/face

mkdir $DATASET_DIR
python3 face_data_downloader.py -d compressed -o $DATASET_DIR --not_altered --not_mask --sample_only
mv $DATASET_DIR/FaceForensics_compressed $FACE_DIR
mv $FACE_DIR/train/original $FACE_DIR/train_videos; rm -rf $FACE_DIR/train
mv $FACE_DIR/val/original $FACE_DIR/val_videos; rm -rf $FACE_DIR/val
mv $FACE_DIR/test/original $FACE_DIR/test_videos; rm -rf $FACE_DIR/test

python3 face_data_processor.py --videos_dir $FACE_DIR
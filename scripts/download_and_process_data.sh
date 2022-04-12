# source $CONDA_PREFIX/etc/profile.d/conda.sh
# conda activate virtual_humans

DATASET_DIR=../datasets
mkdir $DATASET_DIR

FACE_DIR=$DATASET_DIR/face
python3 face_data_downloader.py -d compressed -o $DATASET_DIR --not_altered --not_mask --sample_only
mv $DATASET_DIR/FaceForensics_compressed $FACE_DIR
mkdir $FACE_DIR/train/videos; mv $FACE_DIR/train/original/* $FACE_DIR/train/videos; rm -rf $FACE_DIR/train/original
mkdir $FACE_DIR/validation; mkdir $FACE_DIR/validation/videos; mv $FACE_DIR/val/original/* $FACE_DIR/validation/videos; rm -rf $FACE_DIR/val
mkdir $FACE_DIR/test/videos; mv $FACE_DIR/test/original/* $FACE_DIR/test/videos; rm -rf $FACE_DIR/test/original
python3 face_data_processor.py --videos_dir $FACE_DIR
rm -rf $FACE_DIR/train/videos; rm -rf $FACE_DIR/validation/videos; rm -rf $FACE_DIR/test/videos

wget -O $DATASET_DIR/body.zip "https://www.dropbox.com/s/coapl05ahqalh09/smplpix_data_test_final.zip?dl=1"
unzip $DATASET_DIR/body.zip -d $DATASET_DIR/body_smplpix
mv $DATASET_DIR/body_smplpix/smplpix_data/* $DATASET_DIR/body_smplpix/
rm -rf $DATASET_DIR/body_smplpix/smplpix_data
rm -rf $DATASET_DIR/body.zip
rm -rf $DATASET_DIR/body_smplpix/validation/input/*; rm -rf $DATASET_DIR/body_smplpix/validation/output/*
mv $DATASET_DIR/body_smplpix/train/input/009* $DATASET_DIR/body_smplpix/validation/input/
mv $DATASET_DIR/body_smplpix/train/input/01* $DATASET_DIR/body_smplpix/validation/input/
mv $DATASET_DIR/body_smplpix/train/output/009* $DATASET_DIR/body_smplpix/validation/output/
mv $DATASET_DIR/body_smplpix/train/output/01* $DATASET_DIR/body_smplpix/validation/output/

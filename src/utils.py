import os
import cv2
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from models import Generator, CNNDiscriminator
from vit import create_vit_classifier


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Arguments for running the script")
    parser.add_argument(
        "--datasets_dir",
        type=str,
        required=True,
        # default="/cluster/scratch/aarslan/virtual_humans_data",  # fix
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        # default="/cluster/scratch/aarslan/virtual_humans_data/checkpoints",  # fix
        required=True,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help='Dataset type should be "face" or "body_smplpix".',
        choices=['face', 'body_smplpix']
    )
    parser.add_argument(
        "--discriminator_type",
        type=str,
        choices=['cnn', 'vit', 'mlp-mixer'],
        default='cnn',
        required=True,
    )
    parser.add_argument(
        "--experiment_time",
        type=str,
        default="",
        help="Used in test.py",
    )
    parser.add_argument(
        "--l1_weight",
        type=int,
        default=100,
        help="Weight of l1 loss in generator loss.",
    )
    parser.add_argument("--generator_lr", type=float, default=2e-4)
    parser.add_argument("--discriminator_lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_iterations", type=int, default=200000)

    # VIT
    parser.add_argument("--patch_size", type=int, default=6, help="")
    parser.add_argument("--projection_dim", type=int, default=64, help="")
    parser.add_argument("--norm_eps", type=float, default=1e-6, help="")
    parser.add_argument("--vanilla", dest="vanilla", action="store_true")
    parser.add_argument("--num_heads", type=int, default=4, help="")
    parser.add_argument("--num_transformer_layers", type=int, default=8, help="")
    parser.add_argument("--num_classes", type=int, default=2, help="")

    # FID
    parser.add_argument("--dims", type=int, default=2048, help="")
    parser.add_argument("--num_workers", type=int, default=None, help="")

    return parser


def set_seeds(cfg):
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(cfg["seed"])


def get_dataset(cfg, split):
    input_images_dir = os.path.join(
        cfg["datasets_dir"], cfg["dataset_type"], split, "input"
    )
    input_image_paths = sorted(
        [
            os.path.join(input_images_dir, input_image_name)
            for input_image_name in os.listdir(input_images_dir)
            if input_image_name[-4:] == ".png"
        ]
    )
    real_image_paths = sorted(
        [
            input_image_path.replace("input", "output")
            for input_image_path in input_image_paths
        ]
    )

    ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(input_image_paths),
            tf.data.Dataset.from_tensor_slices(real_image_paths),
        )
    )

    if split == "train":
        ds = ds.map(
            lambda input_image_path, real_image_path: load_images_train(
                input_image_path,
                real_image_path,
                cfg["image_height"],
                cfg["image_width"],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.shuffle(cfg["buffer_size"])
        ds = ds.batch(cfg["batch_size"])
    elif split in ["validation", "test"]:
        ds = ds.map(
            lambda input_image_path, real_image_path: load_images_eval(
                input_image_path,
                real_image_path,
                cfg["image_height"],
                cfg["image_width"],
            )
        )
        ds = ds.batch(cfg["batch_size"])
    else:
        raise Exception(f"Not a valid split {split}.")
    return ds


def get_model(cfg, model_type):
    if model_type == "generator":
        return Generator(cfg)
    elif model_type == "discriminator":
        if cfg["discriminator_type"] == "cnn":
            return CNNDiscriminator(cfg)
        elif cfg["discriminator_type"] == "vit":
            return create_vit_classifier(cfg)
        elif cfg["discriminator_type"] == "mlp-mixer":
            raise NotImplementedError()
        else:
            raise Exception(f"Not a valid discriminator_type {discriminator_type}.")
    else:
        raise Exception(f"Not a valid model_type {model_type}.")


def get_optimizer(cfg, optimizer_type):
    if optimizer_type == "generator":
        return tf.keras.optimizers.Adam(cfg["generator_lr"], beta_1=0.5)
    elif optimizer_type == "discriminator":
        return tf.keras.optimizers.Adam(cfg["discriminator_lr"], beta_1=0.5)
    else:
        raise Exception(f"Not a valid optimizer_type {optimizer_type}.")


def get_time():
    return str(int(time.time()))


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image, image_height, image_width):
    random_start_y = np.random.randint(low=0, high=input_image.shape[0] - image_height)
    random_start_x = np.random.randint(low=0, high=input_image.shape[1] - image_width)

    cropped_input_image = input_image[
                          random_start_y: random_start_y + image_height,
                          random_start_x: random_start_x + image_width,
                          :,
                          ]

    cropped_real_image = real_image[
                         random_start_y: random_start_y + image_height,
                         random_start_x: random_start_x + image_width,
                         :,
                         ]

    return cropped_input_image, cropped_real_image


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1.0
    real_image = (tf.cast(real_image, tf.float32) / 127.5) - 1.0
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image, image_height, image_width):
    input_image, real_image = resize(
        input_image, real_image, int(image_height * 1.15), int(image_width * 1.15)
    )

    input_image, real_image = random_crop(
        input_image, real_image, image_height, image_width
    )

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_images_train(input_image_path, real_image_path, image_height, image_width):
    input_image, real_image = load_images(input_image_path, real_image_path)
    input_image, real_image = random_jitter(
        input_image, real_image, image_height, image_width
    )
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_images_eval(input_image_path, real_image_path, image_height, image_width):
    input_image, real_image = load_images(input_image_path, real_image_path)
    input_image, real_image = resize(input_image, real_image, image_height, image_width)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def load_images(input_image_path, real_image_path):
    # Read and decode an image file to a uint8 tensor
    input_image = tf.io.decode_png(tf.io.read_file(input_image_path))
    real_image = tf.io.decode_png(tf.io.read_file(real_image_path))
    return input_image, real_image


def generator_loss(cfg, disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (cfg["l1_weight"] * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def get_new_directory(folder_names):
    joined_directory = os.path.join(*folder_names)
    os.makedirs(joined_directory, exist_ok=True)
    return joined_directory


def get_checkpoints_dir(cfg):
    return get_new_directory(
        [cfg["checkpoints_dir"], f"experiment_{cfg['experiment_time']}"]
    )


def get_checkpoint_saver(
        cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
):
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
    checkpoint_saver = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    return checkpoint_saver


def save_new_checkpoint(cfg, checkpoint_saver):
    checkpoints_dir = get_checkpoints_dir(cfg)
    # old_checkpoint_file_paths = [
    #     os.path.join(checkpoints_dir, file_name)
    #     for file_name in os.listdir(checkpoints_dir)
    #     if file_name[:4] == "ckpt"
    #     and os.path.isfile(os.path.join(checkpoints_dir, file_name))
    # ]
    # for old_checkpoint_file_path in old_checkpoint_file_paths:
    #     os.system(f"rm -rf {old_checkpoint_file_path}")
    checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
    checkpoint_saver.save(file_prefix=checkpoint_prefix)


def restore_last_checkpoint(cfg, checkpoint_saver):
    checkpoints_dir = get_checkpoints_dir(cfg)
    checkpoint_saver.restore(tf.train.latest_checkpoint(checkpoints_dir))


def get_summary_writer(cfg):
    checkpoints_dir = get_checkpoints_dir(cfg)
    log_dir = get_new_directory([checkpoints_dir, "logs"])
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer


def generate_intermediate_images(cfg, model, test_input, ground_truth, iteration):
    prediction = model(test_input, training=True)
    # Getting the pixel values in the [0, 255] range to plot.
    display_list = [test_input[0], ground_truth[0], prediction[0]]
    file_names = ["input", "ground_truth", "predicted"]

    for i in range(3):
        cv2.imwrite(
            os.path.join(
                get_new_directory(
                    [
                        get_checkpoints_dir(cfg),
                        "intermediate_images",
                        f"iteration_{str(iteration.numpy()).zfill(7)}",
                    ]
                ),
                f"{file_names[i]}.png",
            ),
            ((np.array(display_list[i])[:, :, ::-1] * 0.5 + 0.5) * 255).astype(
                np.int32
            ),
        )


def generate_final_images(cfg, model, test_ds):
    save_idx_counter = 0
    for test_input, _ in test_ds:
        prediction = model(test_input, training=True)
        # Getting the pixel values in the [0, 255] range to plot.
        # prediction = ((prediction.numpy() * 0.5 + 0.5) * 255).astype(np.int32)
        for i in range(prediction.shape[0]):
            cv2.imwrite(
                os.path.join(
                    get_new_directory([get_checkpoints_dir(cfg), "final_images"]),
                    f"{str(save_idx_counter).zfill(7)}.png",
                ),
                ((np.array(prediction[i])[:, :, ::-1] * 0.5 + 0.5) * 255).astype(
                    np.int32
                ),
            )
            save_idx_counter += 1

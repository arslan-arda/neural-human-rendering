import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

from utils import (
    get_argument_parser,
    set_seeds,
    get_dataset,
    get_time,
    get_model,
    get_optimizer,
    get_checkpoint_saver,
    get_manager,
    get_summary_writer,
    generate_final_images,
)

from train import train


if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__

    cfg["mlp_head_units"] = [2048, 1024]
    cfg["transformer_units"] = [cfg["projection_dim"] * 2, cfg["projection_dim"]]

    if cfg["dataset_type"] == "face":
        cfg["num_in_channels"] = 1
        cfg["num_out_channels"] = 3
        cfg["image_height"] = 256
        cfg["image_width"] = 256
    elif cfg["dataset_type"] == "body_smplpix":
        cfg["num_in_channels"] = 3
        cfg["num_out_channels"] = 3
        cfg["image_height"] = 256
        cfg["image_width"] = 256
    else:
        raise Exception(f"Not a valid dataset_type {dataset_type}.")

    cfg["experiment_time"] = get_time()
    set_seeds(cfg)

    train_ds = get_dataset(cfg, split="train")
    val_ds = get_dataset(cfg, split="validation")
    test_ds = get_dataset(cfg, split="test")

    generator = get_model(cfg, model_type="generator")
    discriminator = get_model(cfg, model_type="discriminator")

    generator_optimizer = get_optimizer(cfg, optimizer_type="generator")
    discriminator_optimizer = get_optimizer(cfg, optimizer_type="discriminator")

    checkpoint_saver = get_checkpoint_saver(
        cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
    )
    manager = get_manager(cfg, checkpoint_saver)
    
    summary_writer = get_summary_writer(cfg)

    train(
        cfg,
        generator,
        discriminator,
        generator_optimizer,
        discriminator_optimizer,
        train_ds,
        val_ds,
        summary_writer,
        checkpoint_saver,
        manager
    )

    generate_final_images(cfg, generator, test_ds)

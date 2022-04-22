import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
import os
import time

from utils import (
    generate_intermediate_images,
    generator_loss,
    discriminator_loss,
    save_new_checkpoint,
    get_argument_parser,
    set_seeds,
    get_dataset,
    get_time,
    get_model,
    get_optimizer,
    get_checkpoint_saver,
    get_summary_writer,
    restore_last_checkpoint,
    generate_final_images,
)


@tf.function
def train_step(
    cfg,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    summary_writer,
    iteration,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            cfg, disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=iteration // 1000)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=iteration // 1000)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=iteration // 1000)
        tf.summary.scalar("disc_loss", disc_loss, step=iteration // 1000)
        tf.summary.scalar(
            "disc_mean_real_output", disc_real_output.mean(), step=iteration // 1000
        )
        tf.summary.scalar(
            "disc_mean_generated_output",
            disc_generated_output.mean(),
            step=iteration // 1000,
        )


def train(
    cfg,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    train_ds,
    val_ds,
    summary_writer,
    checkpoint_saver,
):
    example_input, example_target = next(iter(val_ds.take(1)))

    start = time.time()

    for iteration, (input_image, target) in (
        train_ds.repeat().take(cfg["num_iterations"]).enumerate()
    ):
        if (iteration) % 1000 == 0:

            if iteration != 0:
                print(f"Time taken for 10s0 iterations: {time.time()-start:.2f} sec\n")

            start = time.time()

            generate_intermediate_images(
                cfg, generator, example_input, example_target, iteration
            )
            print(f"Iteration: {iteration//1000}k")

        train_step(
            cfg,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            input_image,
            target,
            summary_writer,
            iteration,
        )

        # Training step
        if (iteration + 1) % 10 == 0:
            print(".", end="", flush=True)

        # Save (checkpoint) the model every 5k steps
        if (iteration + 1) % 5000 == 0:
            save_new_checkpoint(cfg, checkpoint_saver)


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
    )

import os
import time
import tensorflow as tf
import tensorboard
from utils import (
    generate_intermediate_images,
    generator_loss,
    discriminator_loss,
    get_checkpoints_dir,
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
    # for e_i, e_t in iter(val_ds.take(1000)):
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
            checkpoints_dir = get_checkpoints_dir(cfg)
            checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
            checkpoint_saver.save(file_prefix=checkpoint_prefix)

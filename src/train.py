import os
import time
import tensorflow as tf
import tensorboard
from utils import (
    generate_intermediate_images,
    generator_loss,
    discriminator_loss,
    save_new_checkpoint,
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
    manager
):
    checkpoint_saver.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

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

        checkpoint_saver.step.assign_add(1)

        # Training step
        if (iteration + 1) % 10 == 0:
            print(".", end="", flush=True)

        if int(checkpoint_saver.step) % 1000 == 0:
            save_path = manager.save()
            print("\n Saved checkpoint for step {}: {} \n".format(int(checkpoint_saver.step), save_path))
            #print("loss {:1.2f}".format(loss.numpy()))

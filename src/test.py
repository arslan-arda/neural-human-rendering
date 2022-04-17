from utils import (
    get_argument_parser,
    set_seeds,
    get_model,
    get_optimizer,
    get_checkpoint_saver,
    restore_last_checkpoint,
    generate_final_images,
)


if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__
    assert (
        cfg["experiment_time"].isdigit() and len(cfg["experiment_time"]) == 10
    ), "experiment_time should be a string of length 10."
    set_seeds(cfg)

    generator = get_model(cfg, model_type="generator")
    discriminator = get_model(cfg, model_type="discriminator")

    generator_optimizer = get_optimizer(cfg, optimizer_type="generator")
    discriminator_optimizer = get_optimizer(cfg, optimizer_type="discriminator")

    checkpoint_saver = get_checkpoint_saver(
        cfg, generator, discriminator, generator_optimizer, discriminator_optimizer
    )

    restore_last_checkpoint(cfg, checkpoint_saver)

    generate_final_images(cfg, generator, test_ds)

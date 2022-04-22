import os
import torch
import pytorch_fid as fid

from utils import get_argument_parser, get_checkpoints_dir

if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__

    if cfg["device"] is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(cfg["device"])

    if cfg["num_workers"] is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = cfg["num_workers"]

    # Ground truth test dataset 
    gt_data = os.path.join(cfg["datasets_dir"], cfg["dataset_type"], "test/output")
    # generated dataset
    gen_data = os.path.join(get_checkpoints_dir(cfg), "final_images")
    paths = [gt_data, gen_data]

    fid_value = fid.calculate_fid_given_paths(paths,
                                          cfg["batch_size"],
                                          device,
                                          cfg["dims"],
                                          num_workers)

    # for now simply print the fid value -> Later we need to store the values somewhere
    print('FID: ', fid_value)
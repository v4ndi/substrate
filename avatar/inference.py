import datetime
import os
import warnings

import hydra
import torch
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from avatar.accelerate_utils import init_accelerate
from avatar.train import log_metrics

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
    f"/home/datalab/nfs/torchinductor_cache/cache_pid{os.getpid()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)


@hydra.main(version_base=None, config_path=".", config_name="inference")
def main(config: DictConfig):
    set_seed(42)
    print(OmegaConf.to_yaml(config))

    model = instantiate(config["model"])
    test_dataloader = instantiate(config["test_dataloader"])

    if "load_state" in config and config["load_state"] is not None:
        load_state = config["load_state"]
        model_state = torch.load(load_state)
        model.load_state_dict(model_state, strict=True)
    else:
        warnings.warn("\n\nNo load_state provided\n\n", stacklevel=2)

    print(model)

    inference(
        config=config,
        model=model,
        test_dataloader=test_dataloader,
    )


def inference(config, model, test_dataloader):
    model.eval()
    mlflow = (
        config["mlflow"]
        if ("mlflow" in config and config["mlflow"] is not None)
        else None
    )
    accelerator = init_accelerate(
        accelerate_arguments=config["accelerator"], mlflow_arguments=mlflow
    )
    if accelerator.num_processes > 1:
        raise NotImplementedError("Distributed inference does not work yet")

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    progress_bar = tqdm(
        test_dataloader,
        desc="Inference: ",
    )

    metrics = instantiate(config["metrics"]["test_metrics"])
    for batch in progress_bar:
        with torch.inference_mode():
            output = model(**batch)
            metrics.update(
                inputs=batch,
                outputs=output,
            )

    scores = metrics.compute()
    if mlflow:
        log_metrics(accelerator, scores, 0, prefix="val_")


if __name__ == "__main__":
    main()

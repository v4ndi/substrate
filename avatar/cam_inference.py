import datetime
import os

import hydra
import torch
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from avatar.accelerate_utils import init_accelerate

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
    f"/home/datalab/nfs/torchinductor_cache/cache_pid"
    f"{os.getpid()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    accelerator = init_accelerate(
        accelerate_arguments=config["accelerator"], mlflow_arguments=None
    )
    set_seed(42)
    accelerator.print(OmegaConf.to_yaml(config))

    model = instantiate(config["model"])
    test_dataloader = instantiate(config["test_dataloader"])
    if "load_state" in config and config["load_state"] is not None:
        load_state = config["load_state"]
        model_state = torch.load(load_state)
        model.load_state_dict(model_state, strict=True)
    else:
        raise ValueError("load state is None")
    accelerator.print(model)
    if accelerator.num_processes > 1:
        raise ValueError("Distributed inference unsupported")

    inference(
        accelerator=accelerator,
        config=config,
        model=model,
        test_dataloader=test_dataloader,
    )


def inference(accelerator, config, model, test_dataloader):
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.eval()
    progress_bar = tqdm(
        test_dataloader,
        desc="Inference: ",
        disable=(not accelerator.is_local_main_process),
    )
    metrics = instantiate(config["metrics"]["test_metrics"])

    for _, batch in enumerate(progress_bar):
        for task_name, task_meta in config["campaign_meta"]["tasks"].items():
            try:
                task_name = [int(task_name)] * len(batch["epk_id"])
                task_name = torch.LongTensor(task_name).to(
                    batch["tab_features"].cat_features.device
                )
            except Exception:
                task_name = [task_name] * len(batch["epk_id"])
            batch["task_name"] = task_name
            for comm_type in task_meta["comm_type"]:
                if comm_type is None:
                    comm_type = -1  # when canal are not supported
                current_comm_type_list = [comm_type] * len(batch["epk_id"])
                batch["target_attr_2"] = current_comm_type_list
                channel_type = torch.LongTensor(current_comm_type_list).to(
                    batch["tab_features"].cat_features.device
                )
                if comm_type is None:
                    batch["group"] = None
                else:
                    batch["group"] = channel_type
                if "is_treat" not in batch:
                    batch["is_treat"] = (
                        torch.ones((len(batch["epk_id"])))
                        .to(batch["tab_features"].cat_features.device)
                        .long()
                    )
                with torch.inference_mode():
                    with accelerator.autocast():
                        output = model(**batch)
                    gathered_objects = accelerator.gather_for_metrics((batch, output))
                    if accelerator.is_main_process:
                        len_gather_obj = len(gathered_objects)
                        assert len_gather_obj % 2 == 0, "incorrect lenght"
                        for n_proc in range(0, len_gather_obj, 2):
                            metrics.update(
                                inputs=gathered_objects[n_proc],
                                outputs=gathered_objects[n_proc + 1],
                            )

    metrics.compute()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()  # noqa

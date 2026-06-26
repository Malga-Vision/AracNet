from typing import Union
import wandb
import torch
import uuid

class WandbWrapper:
    def __init__(self, project_name: str, config: Union[dict, None] = None) -> None:
        self.project_name:  str                 = project_name
        self.run_name:      Union[str, None]    = str(uuid.uuid1())
        self.config:        Union[dict, None]   = config
        self.backend = wandb
        
        self.initialize()


    def initialize(self):
        wandb.init(
            project=self.project_name, 
            name=self.run_name, 
            config=self.config
        )


    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        torch.save(model.state_dict(), f"{model_name}.pth")
        wandb.save(f"{model_name}.pth")
        wandb.log_artifact(f"{model_name}.pth", name=model_name, type="model")

    def log_output(self, output, step=None):
        wandb.log(output, step=step)

    def define_metric(self, metric_name: str, summary: str):
        wandb.define_metric(metric_name, summary=summary)

    def finish(self):
        wandb.finish()


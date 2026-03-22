import torch
from models.optimizer import construct_optimizer


class Adapter:
    def __init__(self, cfg, model, thresholder):
        self.cfg = cfg.clone()
        self.model = model
        self.thresholder = thresholder
        self.cfg.SOLVER = self.cfg.TEST.TTA.SOLVER  # Use TTA solver settings
        self.optimizer = construct_optimizer(self.model, self.cfg)
        self.iter = 0
        self.n_adapt = 0

    @torch.enable_grad()
    def adapt(self, inputs, scores):
        """
        This method is a placeholder for the adapt function.
        It should be overridden in subclasses to implement specific adaptation logic.
        """
        raise NotImplementedError("The adapt method should be implemented in subclasses.")


def construct_adapter(cfg, model, thresholder, **kwargs):
    """
    Constructs an adapter for the given model based on the configuration.
    
    Args:
        cfg: Configuration object containing settings for the adapter.
        model: The model to be adapted.
        
    Returns:
        An instance of Adapter configured with the provided model and settings.
    """
    model_name = cfg.MODEL.NAME
    
    if cfg.TEST.TTA.METHOD == 'M2N2':
        from tta.m2n2.adapter_m2n2 import MLPAdapter, TimesNetAdapter
        cfg.TEST.TTA.STEPS = 1
    elif cfg.TEST.TTA.METHOD == 'CANDI':
        from tta.candi.adapter_candi import MLPAdapter, TimesNetAdapter
    else:
        raise ValueError(f"Unknown TTA method: {cfg.TEST.TTA.METHOD}")
    
    if model_name == "TIMESNET":
        return TimesNetAdapter(cfg, model, thresholder, **kwargs)
    elif model_name == "MLP":
        return MLPAdapter(cfg, model, thresholder, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
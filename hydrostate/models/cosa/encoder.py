"""Shared-LoRA features, with no private exits or extra Landsat input map."""

from hydrostate.registry import MODELS

from ..mtlora import MTLoRAOlmoEarthEncoder


@MODELS.register_module()
class CoSASharedLoRAEncoder(MTLoRAOlmoEarthEncoder):
    def __init__(self, private_rank=0, use_landsat_toa_adapter=False, **kwargs):
        if private_rank != 0 or use_landsat_toa_adapter:
            raise ValueError("CoSA requires shared-only LoRA and no extra TOA input adapter")
        if kwargs.get("finetuning", "lora") != "lora":
            raise ValueError("CoSA encoder uses shared-LoRA adaptation")
        super().__init__(
            tasks=dict(state=dict(support="pixel", grid="output")),
            private_rank=0, use_landsat_toa_adapter=False, **kwargs,
        )

    def forward(self, inputs):
        return super().forward(inputs)["state"]

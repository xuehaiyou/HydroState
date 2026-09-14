data_root = "/fossfs/xiaozhen/HydroState"
manifest = f"{data_root}/manifests/samples.parquet"

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type="default_collate"),
    sampler=dict(type="HydroBalancedSampler", shuffle=True),
    dataset=dict(
        type="HydroStateZarrDataset",
        manifest=manifest,
        data_root=data_root,
        split="train",
        training=True,
        flip_probability=0.5,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type="default_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="HydroStateZarrDataset",
        manifest=manifest,
        data_root=data_root,
        split="validation",
        training=False,
    ),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type="default_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="HydroStateZarrDataset",
        manifest=manifest,
        data_root=data_root,
        split="test",
        training=False,
    ),
)

val_evaluator = dict(type="HydroStateMetric")
test_evaluator = dict(type="HydroStateMetric")


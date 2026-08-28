## Gradient Checkpointing

Тут указано как включить gradient checkpointing в core_fm

### Описание

gradient_checkpointing помогает сэкономить память, путем хранения меньшего количества активаций и их пересчета (вместо хранения) на этапе backward'а

### Как добавить в конфиг
1. Добавить в DistributedDataParallelKwargs асселератору данные 3 строки (некоторые мб лишние, но рекомендуются все, static_graph: True обязателен)
```
accelerator:
    ...
    kwargs_handlers:
    - _target_: accelerate.utils.DistributedDataParallelKwargs
        ...
        find_unused_parameters: False
        broadcast_buffers: False
        static_graph: True
        ...
    ...
```

2. Добавить в GPT2Config строки (обе строки обязательны):
```
_target_: transformers.GPT2Model
config:
  _target_: transformers.GPT2Config
  ...
  ...
  gradient_checkpointing: True
  use_cache: False
  ...
  ...
```


### Пример конфига:
```
accelerator:
  _target_: accelerate.Accelerator
  _partial_: True
  dataloader_config:
    _target_: accelerate.utils.DataLoaderConfiguration
    dispatch_batches: False
  kwargs_handlers:
    - _target_: accelerate.utils.DistributedDataParallelKwargs
      find_unused_parameters: False
      broadcast_buffers: False
      static_graph: True
  gradient_accumulation_steps: 2
  mixed_precision: bf16
train_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.dataset.ShardEventSequenceDataset
    path: /home/datalab/projects/avatards/avatar/core_fm/large/ssl_training/train/large/middle # train
    min_length: 16
    max_length: 512
    random_slicing: True
    event_time_column: "evt_dttm"
    event_ids_column: "event_ids"
    sequence_columns: ['txn_evt_attr_9', 'txn_evt_attr_4', 'txn_evt_attr_5', 'txn_evt_attr_1', 'txn_evt_attr_2', 'txn_evt_attr_3', 'evk_vnv_evt_attr_1', 'evk_vnv_channel_group', 'evk_vnv_sale_product_class_id', 'evk_vnv_sale_type_id', 'pos_geo_evt_attr_1', 'txn_evt_attr_15']
    read_columns: ['event_ids', 'evt_dttm', 'txn_evt_attr_9', 'txn_evt_attr_4', 'txn_evt_attr_5', 'txn_evt_attr_1', 'txn_evt_attr_2', 'txn_evt_attr_3', 'evk_vnv_evt_attr_1', 'evk_vnv_channel_group', 'evk_vnv_sale_product_class_id', 'evk_vnv_sale_type_id', 'pos_geo_evt_attr_1', 'txn_evt_attr_15']
    selected_event_ids: null
    shuffle_files: True
    shuffle_pq: True
  batch_size: 440
  pin_memory: True
  drop_last: True
  num_workers: 8
  collate_fn:
    _target_: avatar.data.dataset.collate_fn.EventSequenceCollateFn
    sequence_columns: ${train_dataloader.dataset.sequence_columns}
    create_attention_mask: True
valid_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.dataset.EventSequenceDataset
    path: /home/datalab/projects/avatards/avatar/core_fm/large/ssl_training/valid
    max_length: 512
    min_length: 1
    event_time_column: "evt_dttm"
    event_ids_column: "event_ids"
    sequence_columns: ['txn_evt_attr_9', 'txn_evt_attr_4', 'txn_evt_attr_5', 'txn_evt_attr_1', 'txn_evt_attr_2', 'txn_evt_attr_3', 'evk_vnv_evt_attr_1', 'evk_vnv_channel_group', 'evk_vnv_sale_product_class_id', 'evk_vnv_sale_type_id', 'pos_geo_evt_attr_1', 'txn_evt_attr_15']
    read_columns: ['event_ids', 'evt_dttm', 'txn_evt_attr_9', 'txn_evt_attr_4', 'txn_evt_attr_5', 'txn_evt_attr_1', 'txn_evt_attr_2', 'txn_evt_attr_3', 'evk_vnv_evt_attr_1', 'evk_vnv_channel_group', 'evk_vnv_sale_product_class_id', 'evk_vnv_sale_type_id', 'pos_geo_evt_attr_1', 'txn_evt_attr_15']
    selected_event_ids: null
    shuffle_files: False
    shuffle_pq: False
  batch_size: 512
  pin_memory: True
  drop_last: False
  num_workers: 8
  collate_fn:
    _target_: avatar.data.dataset.collate_fn.EventSequenceCollateFn
    sequence_columns: ${valid_dataloader.dataset.sequence_columns}
    create_attention_mask: True
model:
  _target_: avatar.pipeline.sequence.NextKTokensPrediction
  enable_event_id_preduction: False
  aggregation_config:
    name: sum
  model:
    _target_: avatar.nn.sequence.TransformersWrapper
    feature_encoder:
      _target_: avatar.nn.feature_encoder.FeatureAttentionEncoder
      dropout_p: 0.1
      time_encoding: delta
      embedding:
        _target_: avatar.nn.embedding.EventSequenceEmbedding
        hidden_size: 128
        columns_meta:
          txn_evt_attr_9:
            type: categorical
            n_classes: 684
            clip: 534
            event_id: [0, 1]
          txn_evt_attr_4:
            type: categorical
            n_classes: 3
            event_id: [0, 1]
          txn_evt_attr_5:
            type: categorical
            n_classes: 3
            event_id: [0, 1]
          txn_evt_attr_1:
            type: categorical
            n_classes: 429
            clip: 400
            event_id: 0
          txn_evt_attr_2:
            type: categorical
            n_classes: 4
            event_id: 0
          txn_evt_attr_3:
            type: categorical
            n_classes: 44
            event_id: 0
          evk_vnv_evt_attr_1:
            type: categorical
            n_classes: 3
            event_id: 2
          evk_vnv_channel_group:
            type: categorical
            n_classes: 10
            event_id: 2
          evk_vnv_sale_product_class_id:
            type: categorical
            n_classes: 37
            event_id: 2
          evk_vnv_sale_type_id:
            type: categorical
            n_classes: 22
            event_id: 2
          pos_geo_evt_attr_1:
            type: categorical
            n_classes: 87
            event_id: null
          txn_evt_attr_15:
            type: numeric
            n_classes: 1
            event_id: [0, 1]
    backbone:
      _target_: transformers.GPT2Model
      config:
        _target_: transformers.GPT2Config
        n_positions: 512
        vocab_size: 0 # может быть любое значение, так как nn.Embedding gpt мы не используем
        n_embd: ${model.model.feature_encoder.embedding.hidden_size}
        n_layer: 6
        n_head: 4
        pad_token_id: 0
        bos_token_id: 1
        eos_token_id: 2
        gradient_checkpointing: True
        use_cache: False

mlflow:
  experiment_name: baselines_seq
  run_name: bs_440_gc_v2
  tracking_uri: /home/datalab/nfs/mlruns_foundation
  logging_dir: /home/datalab/nfs/mlruns_foundation
optimizer:
  _target_: torch.optim.AdamW
  _partial_: True
  lr: 0.001
  weight_decay: 0.01
  scale_lr_multigpu: False
  eps: 1e-4
scheduler:
  _target_: transformers.optimization.get_scheduler
  _partial_: True
  name: cosine
  num_warmup_steps: 6000
  # num_training_steps: null
train:
  start_epoch: 0
  checkpoint_state: null
  model_state: null
  seed: 42
  num_epochs: 40
  clip_grad_norm: 1.5
  max_saved_checkpoints: null
  device_specific: False
  # steps_before_evaluation: 100
metrics:
  valid_metrics:
    _target_: avatar.metrics.MultiLossMetric
  test_metrics:
    _target_: avatar.metrics.CollectEmbeddings
    path_to_save: /home/datalab/nfs/avatar_fm/experiments/meshkovvl/data/embed_collect/baselines_seq/bs_440_gc
    additional_columns: ["report_month", "target_attr_1", "target_attr_2", "target_attr_3", "product_name"]
    save_steps: 250
```

### Результаты:
Значение batch_size указаны так чтобы финальная утилизация составила 38.5 GB на модели из конфига:


| t                           | baseline                       | grad checkpoint wit big batch  |
|-----------------------------|--------------------------------|--------------------------------|
| `batch_size`                | 384                            | 440 (+15%)                     |
| `gradient_checkpointing`    | False                          | True                           |
| `время на эпоху`            | 250 min                        | 270 min (+8%)                  |


## Instruction for MLPBenchmark

Здесь указано как правильно использовать MLP Benchmark

Пример конфига с запуском находится в файле launch_with_MLPBencmark.yaml

Пример конфига для запуска MLP находится в файле MLP.yaml


### Описание

MLPBenchmark нужен для удобного запуска обучения MLP классификации по агрегации hidden и сохранения predict после инференса тестовых данных


### Важные моменты

Важно, чтобы в read_column в test_dataloader были следующие колонки: epk_id, report_month. 

Важно, чтобы в конфиге запуска mlp значение extra_hidden_dim совпадало с размерами hidden, которые приходят на вход после inference тестовых данных.

Важно помнить, что разбиение по датам предполагается по колонке report_month

Важно, чтобы конфиг для запуска MLP был взят из примера.

### Как добавить в конфиг
1. Необходим test_dataloader, затем добавление в test_metrics следующих строчек
```
metrics:
  test_metrics:
    _target_: avatar.metrics.MLPCampaignBenchmark
    path_to_save: /home/datalab/nfs/avatar_fm/experiments/moshcharov/data/final_D_E_B_U_G 
    save_steps: 100 
    path_to_config_dir: /home/datalab/nfs/avatar_fm/experiments/moshcharov/configs 
    config_name: mlp_bench.yaml 
    mlflow_run_name: debug 
    mlflow_exp_name: debug 
    contour_name_column: "product_name" 
    additional_columns: ["target_attr_1", "target_attr_2", "target_attr_3"] 
    split:  
      train: '2025-01-31' 
      valid: '2025-02-28'
      test: '2025-04-31'
    compute_contour: ['cc_response', 'sa_response', 'pl_response', 'td_response']
    cat_feature: True 
```
    path_to_save: - Путь, куда сохрянятся папки для обучения, тренировки и валидации и предикты
    save_steps: - Параметр отвечающий, как часто будут сохраняться parquet файлы
    path_to_config_dir: - Путь до директории с конфигом для запуска пайплайна с классификацией по хидденам
    config_name: - Название конфига из директории, указанной выше
    mlflow_run_name: - run_name для обучения MLP
    mlflow_exp_name: - exp_name для обучения MLP
    contour_name_column: - Название колонки в ваших данных, где указаны контуры, для которых нужно обучение MLP
    additional_columns: - Название колонок, которые вы хотите, чтобы сохранились в parquet кроме epk_id, report_month, seq_hidden_state
    split: - Разбиение данных на соответсвующие выборки по REPORT_MONTH! 
    compute_contour: - Контуры, для которых нужен запуск MLP
    cat_feature: - Флаг для использования target_attr_2, target_attr_3 при обучении MLP головы(при данном флаге нужен конфиг для запуска c этими атрибутами)




2. Необходим конфиг для запуска MLP 


### Пример конфига:
```
PATH_TO_SAVE_PREDICT: /home/datalab/nfs/avatar_fm/experiments/moshcharov/data/debug_cat_feature_mlpbench_rest/debug_mlp_bench_cat_feat_rest_debug/cc_response/predict
accelerator:
  _target_: accelerate.Accelerator
  _partial_: True
  dataloader_config:
    _target_: accelerate.utils.DataLoaderConfiguration
    dispatch_batches: False

  
train_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.dataset.TabularDataset
    path: /home/datalab/nfs/avatar_fm/experiments/moshcharov/data/debug_cat_feature_mlpbench/debug_mlp_bench_cat_feat_debug/cc_response/train
    read_columns: ["seq_hidden_state", "target_attr_1", "target_attr_2", "target_attr_3", "epk_id", 'cat_features']
    shuffle_files: True
    shuffle_pq: True
    hidden_state_column: "seq_hidden_state"
  batch_size: 4096
  pin_memory: True
  drop_last: True
  num_workers: 8
  collate_fn:
    _target_: avatar.data.dataset.collate_fn.TabularCollateFn
    target_column: "target_attr_1"
    
valid_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.dataset.TabularDataset
    path:  /home/datalab/nfs/avatar_fm/experiments/moshcharov/data/debug_cat_feature_mlpbench/debug_mlp_bench_cat_feat_debug/cc_response/valid
    read_columns: ["seq_hidden_state", "target_attr_1", "target_attr_2", "target_attr_3", "epk_id", 'cat_features']
    shuffle_files: false
    shuffle_pq: false
    hidden_state_column: "seq_hidden_state"
  batch_size: 4096
  pin_memory: True
  drop_last: False
  num_workers: 8
  collate_fn:
    _target_: avatar.data.dataset.collate_fn.TabularCollateFn
    target_column: "target_attr_1"
    
test_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: avatar.data.dataset.TabularDataset
    path:  /home/datalab/nfs/avatar_fm/experiments/moshcharov/data/debug_cat_feature_mlpbench/debug_mlp_bench_cat_feat_debug/cc_response/test
    read_columns: ["seq_hidden_state", "target_attr_1", "target_attr_2", "target_attr_3", "epk_id", 'cat_features']
    shuffle_files: false
    shuffle_pq: false
    hidden_state_column: "seq_hidden_state"
  batch_size: 4096
  pin_memory: True
  drop_last: false
  num_workers: 8
  collate_fn:
    _target_: avatar.data.dataset.collate_fn.TabularCollateFn
    target_column: "target_attr_1"    


model:
  _target_: avatar.pipeline.tabular.TabularClassification
  tabular_model:
    _target_: avatar.nn.tabular.MLPEmbedding
    embedding_dim_attr_2: 4
    embedding_dim_attr_3: 2
  num_classes: 2
  dropout_p: 0.15
  task_type: "classification"
  extra_hidden_dim: 128
  out_head_hidden_dim: 512
  hidden_proj_dim: null
  output_head: null
  
mlflow:
  experiment_name: rest_tar_2_tar_3
  run_name: debug
  logging_dir: ./mlruns
  
optimizer:
  _target_: torch.optim.AdamW
  _partial_: True
  lr: 0.0007
  weight_decay: 0.01
  scale_lr_multigpu: True
  eps: 1e-4
  
scheduler:
  _target_: transformers.optimization.get_scheduler
  _partial_: True
  name: cosine
  num_warmup_steps: 6000
  
train:
  start_epoch: 0
  checkpoint_state: null
  model_state: null
  seed: 42
  num_epochs: 50
  clip_grad_norm: 1.5
  max_saved_checkpoints: 15
  device_specific: False
  early_stopping:
    _target_: avatar.train_utils.EarlyStopping
    main_metric: avg_calib_roc_auc_score
    patience: 15
    delta: 0.0001
    strategy: max



metrics:
  valid_metrics:
  - _target_: avatar.metrics.utils.GroupAverageMetricWrapper
    metric:
      _target_: avatar.metrics.utils.GroupAverageMetricWrapper
      metric:
        _target_: avatar.metrics.utils.GroupDevidedMetricsWrapper
        columns_to_devide: ['target_attr_2', 'target_attr_3']
        columns_desc: ['channel', 'group']
        metric_class:
          _target_: avatar.metrics.RocAucScore
          _partial_: true # set true !!!

      avg_over_regulars:
        # calib
        avg_calib_control_group_roc_auc_score: ^channel_\d+_group_1_roc_auc_score
        avg_calib_target_group_roc_auc_score: ^channel_\d+_group_0_roc_auc_score

        avg_calib_channel_0_roc_auc_score: ^channel_0_group_\d+_roc_auc_score
        avg_calib_channel_1_roc_auc_score: ^channel_1_group_\d+_roc_auc_score
        avg_calib_channel_2_roc_auc_score: ^channel_2_group_\d+_roc_auc_score
        avg_calib_channel_3_roc_auc_score: ^channel_3_group_\d+_roc_auc_score

  
    groups: 
      avg_calib_roc_auc_score: ['avg_calib_control_group_roc_auc_score' , 'avg_calib_target_group_roc_auc_score']
    
    
  test_metrics:
  - _target_: avatar.metrics.utils.GroupAverageMetricWrapper
    metric:
      _target_: avatar.metrics.utils.GroupAverageMetricWrapper
      metric:
        _target_: avatar.metrics.utils.GroupDevidedMetricsWrapper
        columns_to_devide: ['target_attr_2', 'target_attr_3']
        columns_desc: ['channel', 'group']
        metric_class:
          _target_: avatar.metrics.RocAucScore
          _partial_: true # set true !!!

      avg_over_regulars:
        # calib
        avg_test_control_group_roc_auc_score: ^channel_\d+_group_1_roc_auc_score
        avg_test_target_group_roc_auc_score: ^channel_\d+_group_0_roc_auc_score

        avg_test_channel_0_roc_auc_score: ^channel_0_group_\d+_roc_auc_score
        avg_test_channel_1_roc_auc_score: ^channel_1_group_\d+_roc_auc_score
        avg_test_channel_2_roc_auc_score: ^channel_2_group_\d+_roc_auc_score
        avg_test_channel_3_roc_auc_score: ^channel_3_group_\d+_roc_auc_score


    groups: 
      avg_calib_roc_auc_score: ['avg_test_control_group_roc_auc_score' , 'avg_test_target_group_roc_auc_score']
      
      avg_test_roc_auc_score: ['avg_test_control_group_roc_auc_score' , 'avg_test_target_group_roc_auc_score']
  - _target_: avatar.post_processing.InferenceCampaign
    path_to_save: ${PATH_TO_SAVE_PREDICT}
    save_steps: 1000
    prefix: null 
```


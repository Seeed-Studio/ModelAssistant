from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTIONS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import Registry

BACKBONES = MMENGINE_MODELS
NECKS = MMENGINE_MODELS
HEADS = MMENGINE_MODELS
LOSSES = MMENGINE_MODELS
POSE_ESTIMATORS = MMENGINE_MODELS

LOOPS = Registry('loop',
                 parent=MMENGINE_LOOPS,
                 locations=['edgelab.engine.runner'])

MODELS = Registry('model',
                  parent=MMENGINE_MODELS,
                  locations=['edgelab.models'])

DATASETS = Registry('dataset',
                    parent=MMENGINE_DATASETS,
                    locations=['edgelab.datasets'])

EVALUATOR = Registry('evaluator',
                     parent=MMENGINE_EVALUATOR,
                     locations=['edgelab.evaluation'])

HOOKS = Registry('hook', parent=MMENGINE_HOOKS, locations=['edgelab.engine'])

DATA_SANPLERS = Registry('data_samplers',
                         parent=MMENGINE_DATA_SAMPLERS,
                         locations=['edgelab.datasets.pipelines'])

METRICS = Registry('metrics',
                   parent=MMENGINE_METRICS,
                   locations=['edgelab.datasets'])

TRANSFORMS = Registry('transforms',
                      parent=MMENGINE_TRANSFORMS,
                      locations=['edgelab.datasets'])

FUNCTIONS = Registry('functions',
                     parent=MMENGINE_FUNCTIONS,
                     locations=['edgelab.datasets'])

_base_ = './pfld_mbv2n_112.py'

num_classes = 4
model = dict(type='PFLD',
             backbone=dict(
                 type='MobileNetV3',
                 inchannel=3,
                 arch='large',
                 out_indices=(3, ),
             ),
             head=dict(type='PFLDhead',
                       num_point=num_classes,
                       input_channel=40,
                       act_cfg="ReLU",
                       loss_cfg=dict(type='L1Loss')))
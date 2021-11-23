from easydict import EasyDict as edict
from pathlib import Path
import torch

def get_config():
    conf = edict()

    # path settings
    # conf.face_root = Path('/data/jhh37/berc/photos_cropped')
    # conf.sketch_root = Path('/data/jhh37/berc/sketches_cropped')
    # conf.pretrained_model_path = 'workspace/save/ms_celeb_1m.pth'
    conf.work_path = Path('workspace/')
    conf.save_path = conf.work_path/'save'
    conf.log_path = conf.work_path / 'log'
    conf.input_size = [256,256]

    # network settings
    conf.embedding_size = 512
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.batch_size = 256
    conf.class_num = 8631

    # optimizer settings
    conf.weight_decay = 5e-4
    conf.lr = 1e-1
    conf.momentum = 0.9

    # others
    conf.milestones = [8,16,21,26]
    conf.num_workers = 8
    conf.pin_memory = True

    return conf


name_dict = {
    '황세례': 'hsr',
    '홍제형': 'hjh',
    '최준용': 'cjy',
    '최익규': 'cik',
    '최원영': 'cwy',
    '최예지': 'cyj',
    '채승호': 'csh',
    '조현우': 'chw',
    '조현용': 'chy',
    '정형주': 'jhj',
    '정제윤': 'jjy',
    '장혁재': 'jhj2',
    '장민성': 'jms',
    '이햇살': 'lhs',
    '이재왕': 'ljw',
    '이석영': 'lsy',
    '이다나': 'ldn',
    '양윤식': 'yys',
    '박현정': 'phj',
    '박주연': 'pjy',
    '박건우': 'pkw',
    '김지수': 'kjs',
    '김영재': 'kyj',
    '김민지': 'kmj',
    '김민수': 'kms',
    '남기표': 'ngp',
    '바린드라': 'brdr',
    '김소영': 'ksy'
}

cfg_re50 = { # configurations for retina fcae used model resnet50
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/media/pc/6T/jasonjzhao/data/faces_emore', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/media/pc/6T/jasonjzhao/buffer/model', # the root to buffer your checkpoints
        LOG_ROOT = '/media/pc/6T/jasonjzhao/buffer/log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = './', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 125, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [35, 65, 95], # epoch stages to decay learning rate

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
),
}

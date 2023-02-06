from yacs.config import CfgNode as CN


_C = CN()

# ========== Dataset & Model configs ==========
_C.DATASET = 'gym99' # 'gym99', 'gym288', 'diving48', 'sthv1'
_C.ARCH = 'tsn_base' # x3d, tsm_resnet18
_C.NUM_FRAMES = 16
_C.NUM_CLASSES = 7 
_C.PRETRAIN_RESNET = True # for resnet
_C.PRETRAIN_PATH_X3D = None # for x3d
_C.INPUT_SIZE = 112 # X3D-M: 224 | X3D-S: 160
_C.SCALE_SIZE = 128 # X3D-M: 256 | X3D-S: 182

# ========== Learning configs ==========
_C.EPOCHS = 40
_C.BATCH_SIZE = 64
_C.LR = 0.002
_C.LR_STEP = [20, 30, 35]
_C.MOMENTUM = 0.9 
_C.WEIGHT_DECAY = 0.0005
_C.CLIP_GRAD = 20 

# ========== Monitor configs ==========
_C.PRINT_FREQ = 50
_C.EVAL_FREQ = 1
_C.EVAL_START = 30

# ========== Runtime configs ==========
_C.SEED = None # Set None if no seed
_C.NUM_WORKERS = 16
_C.GPU_IDS = [0, 1, 2, 3]
_C.ROOT_LOG = 'log'
_C.ROOT_MODEL = 'checkpoint'
_C.STORE_NAME = 'corr_f8'


def get_cfg_defaults():
    return _C.clone()


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    print(cfg.GPU_IDS)
    with open('test.yaml', 'w') as f:
        f.write(cfg.dump())

# Constants for dl_debug (DL_DBG)
DL_DBG = False # if True, debugging will be done
TRAIN_FRAC = 0.8
if DL_TEST:
    DL_DBG_MAX_EPOCHS = 1000
else:
    DL_DBG_MAX_EPOCHS = 1000
DL_DBG_LR = .1
DL_DBG_OVERFIT_EPS_RATIO = .1 # epsilon = 
                          # DL_DBG_OVERFIT_EPS_RATIO*(mean of target)
DL_DBG_BS_PCT = .125 # batch size = floor{(trainset size)*DL_DBG_BS}
DL_DBG_DO_TEST_MEAN = False
DL_DBG_TEST_MEAN_BS = 10
DL_DBG_TEST_MEAN_EPS = 0.1
DL_DBG_IIB_EPOCHS = 100 # IIB = input independent baseline
DL_DBG_IIB_NSHOW = 5
DL_DBG_OVERFIT_BS = 5
DL_DBG_OVERFIT_EPOCHS = 200
DL_DBG_VIS_BS = 124 # VIS = visualize large batch training
DL_DBG_CHART_NSHOW = 5 # CHART = chart dependencies
DL_DBG_CHART_BS = 5
DL_DBG_CMS_NSHOW = 10 # CMS = Choose Model Size by overfit
DL_DBG_NDECIMALS = 4
DL_DBG_SUFFICIENT_R2 = .99
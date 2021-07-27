# Constants for dl_debug (DL_DBG)
RANDOM_SEED = 0
TRAIN_FRAC = 0.8
DL_DBG_MAX_EPOCHS = 100
DL_DBG_LR = .5
DL_DBG_OVERFIT_EPS_RATIO = .05 # epsilon = 
                          # DL_DBG_OVERFIT_EPS_RATIO*(mean of target)
DL_DBG_BS_PCT = .125 # batch size = floor{(trainset size)*DL_DBG_BS}
DL_DBG_DO_TEST_MEAN = False
DL_DBG_TEST_MEAN_BS = 10
DL_DBG_TEST_MEAN_EPS = 0.1
DL_DBG_IIB_EPOCHS = 25 # IIB = input independent baseline
DL_DBG_IIB_THRESHOLD = .75 # the factor, k = (real_data_loss / zero_data_loss), below which the IIB test will pass
DL_DBG_IIB_NSHOW = 5
DL_DBG_OVERFIT_BS = 5
DL_DBG_OVERFIT_EPOCHS = 100
DL_DBG_VIS_BS = 124 # VIS = visualize large batch training
DL_DBG_CHART_NSHOW = 5 # CHART = chart dependencies
DL_DBG_CHART_BS = 5
DL_DBG_CMS_NSHOW = 10 # CMS = Choose Model Size by overfit
DL_DBG_NDECIMALS = 4
DL_DBG_SUFFICIENT_R2 = .99
SAVE_GRAD_IMG = False
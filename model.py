from utils.fastutils import load_learner
model_location = r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\optotaxis calibration\data\segmentation_iteration_testing\models\fastai\iter3_4_nuc_continue.pkl"


if __name__ == "__main__":
    model = load_learner(model_location)

    import numpy as np
    import torch
    inp = np.ndarray((4,3,100,100),dtype='uint8')
    dl = model.dls.test_dl(torch.tensor(inp),num_workers=1)
    # p = model.get_preds(dl=dl,with_decoded=True)
    from IPython import embed; embed()
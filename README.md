# simple-unet-2d
simple unet with NeurIPS'19 topoloss

**Commands:**

* Make sure to populate `train.json` and `test.json` with appropriate hyprerparameters

**Train:**

CUDA_VISIBLE_DEVICES=3 python3 main.py --params ./datalists/DRIVE/train.json
* Ensure `crop_size` in `train.json` is divisible by 16

**Test/Inference:**

CUDA_VISIBLE_DEVICES=4 python3 main.py --params ./datalists/DRIVE/test.json

**Compute Evaluation Metrics (Quantitative Results):**

compute-eval-metrics.py

**Dataset properties:**

GT: Foreground should be 255 ; Background should be 0

* First do pretrain (1000-2000 epochs) by setting `"topo_weight": 0` in `train.json`
* Then, load the best model from pretrain and train using topoloss by setting `topo_weight` to a non-zero value. Change the `output_folder` and `checkpoint_restore` in `train.json` too

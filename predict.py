import os
import cog
import tempfile
from pathlib import Path
import argparse
import os
import numpy as np

from model import CycleGAN
from preprocess import *



class Predictor(cog.Predictor):
    def setup(self):
        """Load models"""

        model_dir_default = './models/sf1_tf2'
        model_name_default = 'checkpoint'
        data_dir_default = './data/evaluation_all/SF1'
        conversion_direction_default = 'A2B'
        output_dir_default = './converted_voices'


        model_dir = './models/sf1_tf2'
        model_name = 'sf1_tf2.ckpt.index'
        #data_dir = argv.data_dir
        conversion_direction = 'A2B'
        #output_dir = argv.output_dir

        num_features = 24
        sampling_rate = 16000
        frame_period = 5.0

        model = CycleGAN(num_features = num_features, mode = 'test')

        model.load(filepath = os.path.join(model_dir, model_name))

        mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
        mcep_mean_A = mcep_normalization_params['mean_A']
        mcep_std_A = mcep_normalization_params['std_A']
        mcep_mean_B = mcep_normalization_params['mean_B']
        mcep_std_B = mcep_normalization_params['std_B']

        logf0s_normalization_params = np.load(os.path.join(model_dir, 'logf0s_normalization.npz'))
        logf0s_mean_A = logf0s_normalization_params['mean_A']
        logf0s_std_A = logf0s_normalization_params['std_A']
        logf0s_mean_B = logf0s_normalization_params['mean_B']
        logf0s_std_B = logf0s_normalization_params['std_B']

    @cog.input("input", type=Path, help="Image path")
    def predict(self, input):
        """Compute prediction"""
        out_path = Path(tempfile.mkdtemp()) / "output.wav"

        print ("coglione di merda")

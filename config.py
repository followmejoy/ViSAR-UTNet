#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py

Set up experiment configuration using argparse library.
"""

import os
import sys
import datetime
import argparse
import numpy as np

def str2bool(v):
    return v.lower() in ( 'true' , '1' )

parser = argparse.ArgumentParser()

# Network arguments
net_arg = parser.add_argument_group ('net')
net_arg.add_argument (
    '-n', '--net', type=str, default='ViSARUTNet',
    help='Network name. e.g. WSA_VSI')
net_arg.add_argument (
    '-T', '--T', type=int, default=4,
    help="Number of layers of LISTA.")
net_arg.add_argument (
    '-p', '--percent', type=float, default=0.8,
    help="Percent of entries to be selected as support in each layer.")
net_arg.add_argument (
    '-maxp', '--max_percent', type=float, default=0.0,
    help="Maximum percentage of entries to be selectedas support in each layer.")
net_arg.add_argument (
    '-l', '--lam', type=float, default=0.4,
    help="Initial lambda in LISTA solvers.")
net_arg.add_argument (
    '-u', '--untied', action='store_true',
    help="Whether weights are untied between layers.")
net_arg.add_argument (
    '-c' , '--coord' , action='store_true',
    help="Whether use independent vector thresholds.")
net_arg.add_argument (
    '-sc', '--scope', type=str, default="",
    help="Scope name of the model.")


# ViSARUTNet Network arguments
ViSARUTNet_arg = parser.add_argument_group ('ViSARUTNet')
ViSARUTNet_arg.add_argument('-d_model', type=int, default=1024)
ViSARUTNet_arg.add_argument('-d_k', type=int, default=128)  
ViSARUTNet_arg.add_argument('-d_v', type=int, default=128)  
ViSARUTNet_arg.add_argument('-d_ff', type=int, default=2048)
ViSARUTNet_arg.add_argument('-n_heads', type=int, default=8)
ViSARUTNet_arg.add_argument('-n_layers', type=int, default=6)
ViSARUTNet_arg.add_argument('-dropout', type=float, default=0.01)
ViSARUTNet_arg.add_argument('-share_proj_weight', action='store_true')
ViSARUTNet_arg.add_argument('-share_embs_weight', action='store_true')
ViSARUTNet_arg.add_argument('-weighted_model', action='store_true')


# Problem arguments
prob_arg = parser.add_argument_group ('prob')
prob_arg.add_argument(
    '-M', '--M', nargs='*', type=int, default=512,
prob_arg.add_argument(
    '-N', '--N', nargs='*', type=int, default=512,
    help="Dimension of sparse codes.")
prob_arg.add_argument(
    '-D', '--D', nargs='*', type=int, default=1,
    help="Dimension of depth.")
prob_arg.add_argument (
    '-sr', '--sample_rate', type=int, default=70,
    help="Sampling rate in compressive sensing experiments.")
prob_arg.add_argument(
    '-S', '--SNR', type=float, default=float('inf'),
    help="Strength of noises.")
prob_arg.add_argument (
    '-task', '--task_type', type=str, default='im',
    help='Task type, in [`im`].')

"""Training arguments."""
train_arg = parser.add_argument_group ('train')
train_arg.add_argument (
    '-lr', '--init_lr', type=float, default=1e-4,
    help="Initial learning rate.")
train_arg.add_argument (
    '-tbs', '--tbs', type=int, default=15,
    help="Training batch size.")
train_arg.add_argument (
    '-vbs', '--vbs', type=int, default=15,
    help="Validation batch size.")
train_arg.add_argument (
    '-fixval', '--fixval', type=str2bool, default=False,
    help="Flag of whether we fix a validation set.")
train_arg.add_argument (
    '-dr', '--decay_rate', type=float, default=0.3,
    help="Learning rate decaying rate after training each layer.")
train_arg.add_argument (
    '-ld', '--lr_decay', type=str, default='0.2,0.02',
    help="Learning rate decaying rate after training each layer.")
train_arg.add_argument (
    '-vs', '--val_step', type=int, default=10,
    help="Interval of validation in training.")
train_arg.add_argument (
    '-mi', '--maxit', type=int, default=12000,
    help="Max number iteration of each stage.")
train_arg.add_argument (
    '-bw', '--better_wait', type=int, default=1000,
    help="Waiting time before jumping to next stage.")


# Experiments arguments
exp_arg = parser.add_argument_group ('exp')
exp_arg.add_argument (
    '-ef', '--exp_folder', type=str, default='./experiments',
    help="Root folder for problems and momdels.")
exp_arg.add_argument (
    '-rf', '--res_folder', type=str, default='./results',
    help="Root folder where test results are saved.")
exp_arg.add_argument (
    '-pf', '--prob_folder', type=str, default='',
    help="Subfolder in exp_folder for a specific setting of problem.")
exp_arg.add_argument (
    '--prob', type=str, default='prob.npz',
    help="Problem file name in prob_folder.")
exp_arg.add_argument (
    '-se', '--sensing', type=str, default=None,
    help="Sensing matrix file. Instance of Problem class.")
exp_arg.add_argument (
    '-dc', '--dict', type=str, default=None,
    help="Dictionary file. Numpy array instance stored as npy file.")
exp_arg.add_argument (
    '-df', '--data_folder', type=str, default=None,
    help="Folder where the tfrecords datasets are stored.")
exp_arg.add_argument (
    '-tf', '--train_file', type=str, default='data/train_data',
    help="File name of tfrecords file of training data for exps.")
exp_arg.add_argument (
    '-vf', '--val_file', type=str, default='data/val_data',
    help="File name of tfrecords file of validation data for exps.")
exp_arg.add_argument (
    '-col', '--column', type=str2bool, default=False,
    help="Flag of whether column-based model is used.")
exp_arg.add_argument (
    '-t' , '--test' , type=str2bool, default=True,
    help="Flag of training or testing models.")
exp_arg.add_argument (
    '-np', '--norm_patch', type=str2bool, default=False,
    help="Flag of normalizing patches in training and testing.")
exp_arg.add_argument (
    '-g', '--gpu', type=str, default='0',
    help="ID's of allocated GPUs.")


def get_config():
    config, unparsed = parser.parse_known_args ()

    """
    Check validity of arguments.
    """
    # check if a network model is specified
    if config.net == None:
        raise ValueError ( 'no model specified' )

    # set experiment path and folder
    if not os.path.exists ( config.exp_folder ):
        os.mkdir ( config.exp_folder )
    if not os.path.exists ( config.res_folder ):
        os.mkdir ( config.res_folder )#'./results'


    """Experiments and results base folder."""
    if config.task_type == 'im':
        # check problem folder: dictionary and sensing matrix
        config.prob_folder = ('im_{}_m{}_n{}_d{}_snr{}_samp{}'.format (
                                config.net, config.M , config.N ,
                                config.D , config.SNR, config.sample_rate ))


    # make experiment base path and results base path
    setattr (config , 'expbase' , os.path.join (config.exp_folder,
                                                config.prob_folder ) )
    setattr (config , 'resbase' , os.path.join (config.res_folder,
                                                config.prob_folder))

    if not os.path.exists (config.expbase):
        os.mkdir (config.expbase)
    if not os.path.exists (config.resbase):
        os.mkdir (config.resbase)


    if config.task_type == 'im':
        # check data files, dictionary and sensing matrix
        if config.train_file is None:
            raise ValueError ("Please provide a training tfrecords file for imaging exp!")
        if config.val_file is None:
            raise ValueError ("Please provide a validation tfrecords file for imaging exp!")

        if not os.path.exists (config.train_file) :
            raise ValueError ('No training data tfrecords file found.')
        if not os.path.exists (config.val_file) :
            raise ValueError ('No validation data tfrecords file found')


    # lr_decay
    config.lr_decay = tuple ([float(decay) for decay in config.lr_decay.split (',')])

    return config, unparsed







# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os


def create_main_output_folder(abs_main_output_folder):
    if not os.path.exists(abs_main_output_folder):
        os.mkdir(abs_main_output_folder)
        print("\t>>Created main output folder: ", abs_main_output_folder)


def create_logs_folder(folder_for_logs):
    if not os.path.exists(folder_for_logs):
        os.mkdir(folder_for_logs)
        print("\t>>Created folder for logs: ", folder_for_logs)


def create_folder_for_predictions(folder_for_predictions):
    if not os.path.exists(folder_for_predictions):
        os.mkdir(folder_for_predictions)
        print("\t>>Created folder for predictions: ", folder_for_predictions)


def create_folder_for_session_results(folder_for_session_results):
    if not os.path.exists(folder_for_session_results):
        os.mkdir(folder_for_session_results)
        print("\t>>Created folder for session: ", folder_for_session_results)


def create_folder_for_segm_and_prob_maps(folder_for_segm_and_prob_maps):
    if not os.path.exists(folder_for_segm_and_prob_maps):
        os.mkdir(folder_for_segm_and_prob_maps)
        print("\t>>Created folder for segmentations and probability maps: ", folder_for_segm_and_prob_maps)


def create_folder_for_features(folder_for_features):
    if not os.path.exists(folder_for_features):
        os.mkdir(folder_for_features)
        print("\t>>Created folder for features: ", folder_for_features)


def make_folders_for_test_session(abs_main_output_folder, session_name):
    # Create folders for saving the prediction images:
    print("Creating necessary folders for testing session...")
    create_main_output_folder(abs_main_output_folder)

    folder_for_logs = abs_main_output_folder + "/logs/"
    create_logs_folder(folder_for_logs)

    folder_for_predictions = abs_main_output_folder + "/predictions"
    create_folder_for_predictions(folder_for_predictions)

    folder_for_session_results = folder_for_predictions + "/" + session_name
    create_folder_for_session_results(folder_for_session_results)

    folder_for_segm_and_prob_maps = folder_for_session_results + "/predictions/"
    create_folder_for_segm_and_prob_maps(folder_for_segm_and_prob_maps)

    folder_for_features = folder_for_session_results + "/features/"
    create_folder_for_features(folder_for_features)

    return [folder_for_logs, folder_for_segm_and_prob_maps, folder_for_features]


def create_folder_for_cnn_models(folder_for_cnn_models):
    if not os.path.exists(folder_for_cnn_models):
        os.mkdir(folder_for_cnn_models)
        print("\t>>Created folder to save cnn-models as they get trained: ", folder_for_cnn_models)


def create_folder_for_session_cnn_models(folder_for_session_cnn_models):
    if not os.path.exists(folder_for_session_cnn_models):
        os.mkdir(folder_for_session_cnn_models)
        print("\t>>Created folder to save session's cnn-models as they get trained: ", folder_for_session_cnn_models)


def create_folder_for_tensorboard(folder_for_tensorboard):
    if not os.path.exists(folder_for_tensorboard):
        os.mkdir(folder_for_tensorboard)
        print("\t>>Created folder to log tensorboard metrics/events: ", folder_for_tensorboard)


def create_folder_for_session_tensorboard(folder_for_session_tensorboard):
    if not os.path.exists(folder_for_session_tensorboard):
        os.mkdir(folder_for_session_tensorboard)
        print("\t>>Created folder to log session's tensorboard metrics/events: ", folder_for_session_tensorboard)


def make_folders_for_train_session(abs_main_output_folder, session_name):
    # Create folders for saving the prediction images:
    print("Creating necessary folders for training session...")
    create_main_output_folder(abs_main_output_folder)

    folder_for_logs = abs_main_output_folder + "/logs/"
    create_logs_folder(folder_for_logs)

    folder_for_cnn_models = abs_main_output_folder + "/saved_models/"
    create_folder_for_cnn_models(folder_for_cnn_models)
    folder_for_session_cnn_models = folder_for_cnn_models + "/" + session_name + "/"
    create_folder_for_session_cnn_models(folder_for_session_cnn_models)

    folder_for_predictions = abs_main_output_folder + "/predictions"
    create_folder_for_predictions(folder_for_predictions)
    folder_for_session_results = folder_for_predictions + "/" + session_name
    create_folder_for_session_results(folder_for_session_results)
    folder_for_segm_and_prob_maps = folder_for_session_results + "/predictions/"
    create_folder_for_segm_and_prob_maps(folder_for_segm_and_prob_maps)
    folder_for_features = folder_for_session_results + "/features/"
    create_folder_for_features(folder_for_features)

    folder_for_tensorboard = abs_main_output_folder + "/tensorboard/"
    create_folder_for_tensorboard(folder_for_tensorboard)
    folder_for_session_tensorboard = folder_for_tensorboard + "/" + session_name + "/"
    create_folder_for_session_tensorboard(folder_for_session_tensorboard)

    return [
        folder_for_logs,
        folder_for_session_cnn_models,
        folder_for_segm_and_prob_maps,
        folder_for_features,
        folder_for_session_tensorboard,
    ]


def make_folders_needed_for_create_model_session(abs_main_output_folder, model_name):
    # Create folders for saving the prediction images:
    print("Creating necessary folders for create-new-model session...")
    create_main_output_folder(abs_main_output_folder)

    folder_for_logs = abs_main_output_folder + "/logs/"
    create_logs_folder(folder_for_logs)

    folder_for_cnn_models = abs_main_output_folder + "/saved_models/"
    create_folder_for_cnn_models(folder_for_cnn_models)

    folder_for_session_cnn_models = folder_for_cnn_models + "/" + model_name + "/"
    create_folder_for_session_cnn_models(folder_for_session_cnn_models)

    return [folder_for_logs, folder_for_session_cnn_models]


def handle_exception_tf_restore(log, exc):
    import sys, traceback

    log.print3("")
    log.print3(
        "ERROR: DeepMedic caught exception when trying to load parameters from the given path of a previously saved model.\n"
        + "Two reasons are very likely:\n"
        + "a) Most probably you passed the wrong path. You need to provide the path to the Tensorflow checkpoint, as expected by Tensorflow.\n"
        + "\t In the traceback further below, Tensorflow may report this error of type [NotFoundError].\n"
        + "\t DeepMedic uses tensorflow checkpoints to save the models. For this, it stores different types of files for every saved timepoint.\n"
        + "\t Those files will be by default in ./examples/output/saved_models, and of the form:\n"
        + "\t filename.datetime.model.ckpt.data-0000-of-0001 \n"
        + "\t filename.datetime.model.ckpt.index \n"
        + "\t filename.datetime.model.ckpt.meta (Maybe this is missing. That's ok.) \n"
        + "\t To load this checkpoint, you have to provide the path, OMMITING the part after the [.ckpt]. I.e., your command should look like:\n"
        + "\t python ./deepMedicRun.py -model path/to/model/config -train path/to/train/config -load filename.datetime.model.ckpt \n"
        + "b) You have created a network of different architecture than the one that is being loaded and Tensorflow fails to match their variables.\n"
        + "\t If this is the case, Tensorflow may report it below as error of type [DataLossError]. \n"
        + "\t If you did not mean to change architectures, ensure that you point to the same modelConfig.cfg as used when the saved model was made.\n"
        + "\t If you meant to change architectures, then you will have to create your own script to load the parameters from the saved checkpoint,"
        + " where the script must describe which variables of the new model match the ones from the saved model.\n"
        + 'c) The above are "most likely" reasons, but others are possible.'
        + " Please read the following Tensorflow stacktrace and error report carefully, and debug accordingly...\n"
    )
    log.print3(traceback.format_exc())
    sys.exit(1)

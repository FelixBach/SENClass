from datetime import datetime
import geodata
import sample_selection
import random_forest
import numpy as np
import os
import AccuracyAssessment
import matplotlib.pyplot as plt

start_time = datetime.now()


def main():
    #####     INPUTS     #####
    # path = "D:/Uni/GEO419/T2/Abschlussaufgabe/Spain_Donana_S1-VV/"
    path = "C:/GEO419/Spain_Donana_S1-VV/"
    # path = "/home/felix/PycharmProjects/SENClass/test_data/s1/S1_resamp/"
    # path_ref_p = "/home/felix/PycharmProjects/SENClass/test_data/clc/"

    raster_ext = "tif"
    out_folder_resampled_scenes = "test/"

    # path_ref_p = "D:/Uni/GEO419/T2/Abschlussaufgabe/"  # path to reference product
    path_ref_p = "C:/GEO419/"  # path to reference product

    # file_name = "S1A__IW___A_20180620T182625_147_VV_grd_mli_norm_geo_db_resampled.tif"
    ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_reprojected.tif"

    # inputs for sample selection
    train_size = 0.25   # between 0 and 1.0; represent the proportion of the dataset to include in the train split
    random_state = np.random.randint(low=0, high=43)   # random value for sample selection and random forest
    strat = True   # True: using stratified random sampling, False: using random sampling

    # inputs for random forrest algorithm
    max_depth = 5   # The maximum depth of the tree, default none
    n_estimator = 10  # The number of trees in the forest, default 100

    #####     FUNCTIONS     #####
    # do stuff with geodata
    geodata.reproject(path, path_ref_p, ref_p_name, raster_ext, out_folder_resampled_scenes)

    # select samples from scene(s)
    x_train, x_test, y_train, y_test = sample_selection.select_samples(path_ref_p, ref_p_name, path,
                                                                       out_folder_resampled_scenes, raster_ext,
                                                                       train_size, random_state, strat)

    # do random forrest stuff
    pred = random_forest.random_forest_classifier(max_depth, n_estimator, random_state, x_train, x_test, y_train)

    # get accuracy and other metrics
    AccuracyAssessment.accuracy(pred, y_test)  # get overall accuracy
    # confusion_matrix = AccuracyAssessment.get_confusion_matrix(y_test, pred)  # get confusion matrix
    # AccuracyAssessment.plot_confusion_matrix(confusion_matrix)  # heatmap of Confusion Matrix
    # AccuracyAssessment.get_kappa(confusion_matrix)  # get Cappa Coefficient
    # plt.show()

    end_time = datetime.now()
    print(f"\n end-time =", end_time - start_time, "Hr:min:sec \n")


# main func
if __name__ == '__main__':
    main()

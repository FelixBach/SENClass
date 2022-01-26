from datetime import datetime
import geodata
import sample_selection
import random_forest
import numpy as np
import AccuracyAssessment
import matplotlib.pyplot as plt

start_time = datetime.now()


def main():
    #####     INPUTS     #####
    # path = "D:/Uni/GEO419/T2/Abschlussaufgabe/Spain_Donana_S1-VV/"
    path = "C:/GEO419/test_env/s1/"
    # path = "/home/felix/PycharmProjects/SENClass/test_data/s1/"
    # path_ref_p = "/home/felix/PycharmProjects/SENClass/test_data/clc/"

    raster_ext = "tif"
    out_folder_resampled_scenes = "resamp/"
    # out_folder_resampled_scenes = "test/"
    # out_folder_resampled_scenes = "S1_resamp/"

    # path_ref_p = "D:/Uni/GEO419/T2/Abschlussaufgabe/"  # path to reference product
    path_ref_p = "C:/GEO419/test_env/"  # path to reference product

    # file_name = "S1A__IW___A_20180620T182625_147_VV_grd_mli_norm_geo_db_resampled.tif"
    # ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_reprojected_reprojected_3classs.tif"
    # ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_reprojected_reprojected.tif"
    # ref_p_name = "CLC_subset_reclass_reprojected.tif"
    ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_wgs.tif"  # linux

    out_folder_prediction = "results/"    # path from output folder
    name_predicted_image = "prediction_1"

    # random_state = np.random.randint(low=0, high=43)  # random value for sample selection and random forest
    random_state = 0

    # inputs for sample_selection.select_samples
    train_size = 0.25  # Specifies how many samples are used for training
    strat = False  # True: using stratified random sampling, False: using random sampling

    # random forest parameter
    max_depth = 2  # The maximum depth of the tree, default none
    n_estimator = 3  # The number of trees in the forest, default 100
    n_cores = -1  # defines number of cores to use, if -1 all cores are used
    verbose = 2  # shows output from random forrest in console

    #####     FUNCTIONS     #####
    # reprojecting and reclassifying raster data
    out_ref_p = geodata.reproject_raster(path, path_ref_p, ref_p_name, raster_ext, out_folder_resampled_scenes)
    geodata.reclass_raster(out_ref_p)

    # select samples from scene(s)
    x_train, x_test, y_train, y_test, data = sample_selection.select_samples(path, path_ref_p, out_ref_p,
                                                                             out_folder_resampled_scenes, raster_ext,
                                                                             train_size, random_state, strat)

    # train random forest and predict result
    rf, rf_fitted = random_forest.rf_fit(max_depth, random_state, n_estimator, n_cores, verbose, x_train, y_train)
    prediction = random_forest.rf_predict(data, rf_fitted)
    geodata.prediction_to_gtiff(prediction, path, out_folder_prediction, name_predicted_image, out_ref_p, raster_ext)

    # get accuracy and other metrics
    AccuracyAssessment.accuracy(prediction, out_ref_p)  # get overall accuracy
    confusion_matrix = AccuracyAssessment.get_confusion_matrix(prediction, out_ref_p)  # get confusion matrix
    AccuracyAssessment.plot_confusion_matrix(confusion_matrix)  # heatmap of Confusion Matrix
    AccuracyAssessment.get_kappa(confusion_matrix)  # get Cappa Coefficient

    end_time = datetime.now()
    print(f"\n end-time =", end_time - start_time, "Hr:min:sec \n")

    plt.show()


# main func
if __name__ == '__main__':
    main()

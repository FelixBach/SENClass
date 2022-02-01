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
    # path = "C:/GEO419/test_env/s1/"
    path = "D:/Uni/GEO419/T2/Abschlussaufgabe/Spain_Donana_S1-VV/"
    path = "/home/felix/Dokumente/SENClass/test_env/10_files/"
    #path = "C:/GEO419/test_env/s1/"

    raster_ext = "tif"
    out_folder_resampled_scenes = "resamp/"
    # out_folder_resampled_scenes = "test/"
    # out_folder_resampled_scenes = "S1_resamp/"

    # path_ref_p = "D:/Uni/GEO419/T2/Abschlussaufgabe/"  # path to reference product
    path_ref_p = "C:/GEO419/test_env/"  # path to reference product
    path_ref_p = "/home/felix/Dokumente/SENClass/test_env"

    # ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_reprojected_reprojected_3classs.tif"
    # ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_reprojected_reprojected.tif"
    ref_p_name = "seasonality_10W_40Nv1_3_2020_sub_wgs.tif"
    # path_ref_p = "C:/GEO419/test_env/"  # path to reference product

    out_folder_prediction = "results/"  # path from output folder
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
    verbose = 1  # shows output from random forrest in console

    # random forest tuning parameter
    min_depth_t = 2
    max_depth_t = 5
    min_estimator = 10  # minimum number of estimators
    max_estimator = 20  # maximum number of estimators
    number_estimator = 2  # Number of samples to generate
    n_iter = 2  # Number of parameter settings that are sampled
    cv = 2  # number of folds of cross validation

    #####     FUNCTIONS     #####
    # reprojecting and reclassifying raster data
    out_ref_p = geodata.reproject_raster(path, path_ref_p, ref_p_name, raster_ext, out_folder_resampled_scenes)
    geodata.reclass_raster(out_ref_p)

    # select samples from scene(s)
    x_train, x_test, y_train, y_test, data, mask = sample_selection.select_samples(path, path_ref_p, out_ref_p,
                                                                             out_folder_resampled_scenes, raster_ext,
                                                                             train_size, random_state, strat)

    # create random forest
    rf = random_forest.rf_create(max_depth, random_state, n_estimator, n_cores, verbose)

    # train random forest
    rf_fitted = random_forest.rf_fit(rf, x_train, y_train)
    
    # implement PCA Transformation
    #data, x_train = random_forest.principal(data, x_train)
    prediction = random_forest.rf_predict(data, rf_fitted)

    print("Acc for base model")
    AccuracyAssessment.accuracy(prediction, out_ref_p)  # get overall accuracy
    base_matrix = AccuracyAssessment.get_confusion_matrix(prediction, out_ref_p)  # get confusion matrix
    AccuracyAssessment.plot_confusion_matrix(base_matrix)  # heatmap of Confusion Matrix
    AccuracyAssessment.get_kappa(base_matrix)

    geodata.prediction_to_gtiff(prediction, path, out_folder_prediction, name_predicted_image, out_ref_p, raster_ext, mask)

    tuned_prediction = random_forest.rf_parameter_tuning(x_train, y_train, data, min_depth_t, max_depth_t, min_estimator,
                                                         max_estimator, number_estimator, n_iter, cv, random_state,
                                                         n_cores)

    print("Acc for tuned model")
    # get accuracy and other metrics
    AccuracyAssessment.accuracy(tuned_prediction, out_ref_p)  # get overall accuracy
    tuned_matrix = AccuracyAssessment.get_confusion_matrix(tuned_prediction, out_ref_p)  # get confusion matrix
    AccuracyAssessment.plot_confusion_matrix(tuned_matrix)  # heatmap of Confusion Matrix
    AccuracyAssessment.get_kappa(tuned_matrix)  # get Cappa Coefficient

    end_time = datetime.now()
    print(f"\n end-time =", end_time - start_time, "Hr:min:sec \n")

    # plt.show()


# main func
if __name__ == '__main__':
    main()
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from SENClass import geodata
from SENClass import random_forest
from SENClass import accuracy_assessment
from SENClass import pca

start_time = datetime.now()


def main():
    #####     INPUTS     #####
    path = "C:/GEO419/Spain_Donana_S1-VV/"

    out_folder_resampled_scenes = "resamp/"
    path_ref_p = "C:/GEO419/"  # path to reference product

    ref_p_name = "CLC_subset.tif" #
    raster_ext = "tif"

    out_folder_prediction = "results/"  # path from output folder
    name_predicted_image = "base_prediction_nd_0"
    name_tuned_predicted_image = "tune_prediction_nd_0"

    # values for reclassifying raster data
    raster_value = [0, 11, 12]
    class_value = [100, 200, 300]

    # random_state = np.random.randint(low=0, high=43)  # random value for sample selection and random forest
    random_state = 0

    # inputs for sample_selection.select_samples
    train_size = 0.25   # Specifies how many samples are used for training
    sss = False  # True: using StratifiedShuffleSplit, False: using train_test_split for sampling

    # pca
    n_components = 0.95

    # random forest parameter
    max_depth = 5   # The maximum depth of the tree, default none
    n_estimator = 15  # The number of trees in the forest, default 100
    n_cores = -1  # defines number of cores to use, if -1 all cores are used
    verbose = 1  # shows output from random forrest in console

    # random forest tuning parameter
    min_depth_t = 3
    max_depth_t = 10
    min_estimator = 10  # minimum number of estimators
    max_estimator = 50  # maximum number of estimators
    value_generator = 5  # Number of values to geneurate
    n_iter = 5  # Number of parameter settings that are sampled

    #####     FUNCTIONS     #####
    # reprojecting and reclassifying raster data
    out_ref_p = geodata.reproject_raster(path, path_ref_p, ref_p_name, raster_ext, out_folder_resampled_scenes)
    # reclass raster file
    geodata.reclass_raster(raster_value, class_value, out_ref_p)

    # select samples from scene(s)
    x_train, y_train, data, mask = geodata.select_samples(path, path_ref_p, out_ref_p, out_folder_resampled_scenes,
                                                          raster_ext, train_size, random_state, sss)
    # implement PCA Transformation
    # data, x_train = pca.principal(data, x_train, n_components)

    # create random forest
    rf = random_forest.rf_create(max_depth, random_state, n_estimator, n_cores, verbose)

    # train random forest
    rf_fitted = random_forest.rf_fit(rf, x_train, y_train)

    # predict result
    prediction = random_forest.rf_predict(data, rf_fitted)

    # calculate Accuracy
    accuracy_assessment.accuracy_assessment(prediction, out_ref_p)
    geodata.prediction_to_gtiff(prediction, path, out_folder_prediction, name_predicted_image, out_ref_p, raster_ext,
                                mask)
    # visualize the results
    geodata.tif_visualize(path, out_folder_prediction, name_predicted_image, raster_ext)

    # parameter tuning
    tuned_prediction = random_forest.rf_parameter_tuning(x_train, y_train, data, min_depth_t, max_depth_t,
                                                         min_estimator, max_estimator, value_generator, n_iter,
                                                         random_state, n_cores)

    geodata.prediction_to_gtiff(tuned_prediction, path, out_folder_prediction, name_tuned_predicted_image, out_ref_p,
                                raster_ext, mask)
    # visualize the results
    geodata.tif_visualize(path, out_folder_prediction, name_tuned_predicted_image, raster_ext)

    # calculate Accuracy
    accuracy_assessment.accuracy_assessment(tuned_prediction, out_ref_p)

    end_time = datetime.now()
    print(f"\n end-time =", end_time - start_time, "Hr:min:sec \n")

    # plt.show()


# main func
if __name__ == '__main__':
    main()

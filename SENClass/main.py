from datetime import datetime
import geodata
import sample_selection
import random_forrrest
import numpy as np

start_time = datetime.now()


def main():
    # path = "C:/GEO419/Spain_Donana_S1-VV/"
    path = "C:/GEO419/Spain_Donana_S1-VV/test_data/"
    # path = "/home/felix/PycharmProjects/SENClass/test_data/s1/S1_resamp/"
    file_name = "S1A__IW___A_20180608T182624_147_VV_grd_mli_norm_ge_resamp_100m.tif"
    raster_ext = "tif"
    path_clc = "C:/GEO419/"
    # path_clc = "/home/felix/PycharmProjects/SENClass/test_data/clc/"
    clc_name = "CLC_subset.tif"

    # do stuff with geodata
    raster_file_list, raster_file_name = geodata.parse_folder(path, raster_ext)
    clc_reclass = geodata.reclass_clc(path_clc, clc_name)
    geodata.reproject(path, clc_reclass, raster_file_list, raster_file_name)

    # select samples from scene(s)
    x_train, x_test, y_train, y_test = sample_selection.select_samples(path_clc, path, file_name)

    # do random forrest stuff
    # random_state = np.random.randint(low=75, high=100)  # set min and max value for random, maybe needed for rf
    # classifier
    rfcb = random_forrrest.rf_classifier_basic(max_depth=5, n_estimator=5)  # train rf
    rfcb_fit = random_forrrest.rf_classifier_fit(rfcb, x_train, y_train)    # fit model
    pred = random_forrrest.rf_calssifier_predict(rfcb_fit, x_test)  # get result

    random_forrrest.accu(pred, y_test)  # get overall accuracy

    end_time = datetime.now()
    print(f"\n end-time =", end_time - start_time, "Hr:min:sec \n")


# main func
if __name__ == '__main__':
    main()

from datetime import datetime
import geodata
import random_forrrest

start_time = datetime.now()


def main():
    path = "C:/GEO419/Spain_Donana_S1-VV/"
    file_name = "S1A__IW___A_20180103T182622_147_VV_grd_mli_norm_geo_db.tif"
    raster_ext = "tif"
    path_clc = "C:/GEO419/CLC_subset.tif"

    # do stuff with geodata
    # geodata.parse_folder(path, raster_ext)
    # geodata.open_raster(path, file_name)
    geodata.adjust_clc(path_clc)

    # do random forrest stuff
    # random_forrrest.rf_classifier_basic(max_depth=5, n_estimator=5)  # 5 just for testing

    end_time = datetime.now()
    print(f"\n end-time =", end_time - start_time, "Hr:min:sec \n")


# main func
if __name__ == '__main__':
    main()

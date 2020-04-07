import os
import sys
import shutil
import pandas as pd
import pandas.io.common
import pyarrow as pa
import pyarrow.parquet as pq
import tarfile

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False


def run_conversion(path, out_formats):
    files = [os.path.join(path, f) for f in ["hubb.andpar","hubb.dat","split_files.sh","chi_asympt", "F_DM", "GAMMA_DM_FULLRANGE", "vert_chi","g0m", "g0mand", "gm_wim"]]
    dirs = [os.path.join(path,f) for f in ["trilex_omega", "tripamp_omega", "trip_omega"]]
    files_not_found = []

    def read_and_convert(path, fn, header_names, output_formats):
        fn = os.path.join(path,fn)
        if len(output_formats):
            if "hdf5" in output_formats or "parquet" in output_formats:
                df = pd.read_csv(fn, sep=' ', skipinitialspace=True, header=None, names=header_names)
                df = df[df.applymap(isnumber)]
                df = df.applymap(float)
            if "hdf5" in output_formats:
                df.to_hdf('results.h5', key=fn, mode='w')
            if "parquet" in output_formats:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, fn+'.parquet')
        else:
            print("Warning: empty list of output formats!")

    def read_dir_and_convert(path, dn, header_names, output_formats):
        dn = os.path.join(path,dn)
        dfi = {}
        if "hdf5" in output_formats or "parquet" in output_formats:
            for fn in os.listdir(dn):
                omega = int(fn[-3:])
                dfi[fn[-3:]] = pd.read_csv(os.path.join(dn,fn), sep=' ', skipinitialspace=True, header=None, names=header_names)
            df = pd.concat(dfi)
            df = df[df.applymap(isnumber)]
            df = df.applymap(float)

        if "hdf5" in output_formats:
            df.to_hdf('results.h5', key=dn, mode='w')
        if "parquet" in output_formats:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, dn+'.parquet')


    # =========================================================================== 
    # =                             chi_asympt                                  =
    # =========================================================================== 
    fn = "chi_asympt"
    hn = ["omega","Re_chi_up", "Im_chi_up", "Re_chi_do", "Im_chi_do", "Re_chi_ch", "Im_chi_ch"]
    print("storing " + fn)
    try:
        read_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_n.append(fn)

    # =========================================================================== 
    # =                                F_DM                                     =
    # =========================================================================== 
    fn = "F_DM"
    hn = ["Re_F_density", "Im_F_density", "Re_F_magnetic", "Im_F_magnetic"]
    print("storing " + fn)
    try:
        read_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_n.append(fn)


    # =========================================================================== 
    # =                         GAMMA_DM_FULLRANGE                              =
    # =========================================================================== 
    fn = "GAMMA_DM_FULLRANGE"
    hn = ["Re_Gamma_density", "Im_Gamma_density", "Re_Gamma_magnetic", "Im_Gamma_magnetic"]
    print("storing " + fn)
    try:
        read_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_not_found.append(fn)

    # =========================================================================== 
    # =                              vert_chi                                   =
    # =========================================================================== 
    fn = "vert_chi"
    hn = ["Re_Chi_density", "Im_Chi_density", "Re_Chi_magnetic", "Im_Chi_magnetic"]
    print("storing " + fn)
    try:
        read_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_not_found.append(fn)


    # =========================================================================== 
    # =                          g0m,g0man,gm_wim                               =
    # =========================================================================== 
    fns = ["g0m", "g0mand", "gm_wim"]
    print("storing " + str(fns))
    hn = ["Re", "Im"]
    for fn in fns:
        try:
            read_and_convert(path, fn, hn, out_formats)
        except pandas.io.common.EmptyDataError:
            print("WARNING: "+fn+" not found!")
            files_not_found.append(fn)

    # =========================================================================== 
    # =                               Trilex                                    =
    # =========================================================================== 
    fn = "tripamp_omega"
    print("storing " + fn)
    hn = ["Re_TripAmp_density", "Im_TripAmp_density", "Re_TripAmp_magnetic",\
            "Im_TripAmp_magnetic", "Re_TripAmp_pp", "Im_TripAmp_pp"]
    try:
        read_dir_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_not_found.append(fn)

    fn = "trip_omega"
    print("storing " + fn)
    hn = ["Re_Trip_up", "Im_Trip_up", "Re_Trip_do", "Im_Trip_do", "Re_Trip_pp", "Im_Trip_pp"]
    try:
        read_dir_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_not_found.append(fn)

    fn = "trilex_omega"
    print("storing " + fn)
    hn = ["Re_Trilex_density", "Im_Trilex_density", "Re_Trilex_magnetic",\
            "Im_Trilex_magnetic", "Re_Trilex_pp", "Im_Trilex_pp"]
    try:
        read_dir_and_convert(path, fn, hn, out_formats)
    except pandas.io.common.EmptyDataError:
        print("WARNING: "+fn+" not found!")
        files_not_found.append(fn)

    if "tar" in out_formats:
        print("creating tar.bz2 archive. This can take several minutes.")
        tar = tarfile.open("results.tar.bz2", "w:bz2")
        for name in files:
            print(" ".ljust(84))
            print(("\rtaring: " + str(name)), end='')
            tar.add(name)
        for name in dirs:
            print(" ".ljust(84))
            print(("\rtaring: " + str(name)), end='')
            tar.add(name)
        tar.close()
        print("\nTarfile Created.")

    if "text" not in out_formats:
        print("deleting old remaining csv files")
        for filename in (files+dirs):
            try:
                if os.path.isfile(filename) or os.path.islink(filename):
                    os.unlink(filename)
                elif os.path.isdir(filename):
                    shutil.rmtree(filename)
            except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    return files_not_found

if __name__ == "__main__":
    if len(sys.argv)==1:
        path = os.path.abspath(".")
        out_formats = ["text","tar","hdf5","parquet"]
        run_conversion(path, out_formats)
    elif len(sys.argv)==2:
        path = os.path.abspath(sys.argv[1])
        out_formats = ["text","tar","hdf5","parquet"]
        run_conversion(path, out_formats)
    else:
        path = os.path.abspath(sys.argv[1])
        out_formats = (sys.argv[2]).split(',')
        run_conversion(path, out_formats)

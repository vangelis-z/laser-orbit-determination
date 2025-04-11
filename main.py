# -*- coding: utf-8 -*-


import datetime
import os

import jdk4py
import orekit_jpype as orekit

# Initializing Orekit and JVM
if "JAVA_HOME" not in os.environ:
    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
orekit.initVM()
orekit.pyhelpers.setup_orekit_data(from_pip_library=True)

import slrDataUtils as slr


USER = 'vanzach@mail.ntua.gr'

REF_TIME = datetime.datetime(2015, 12, 15)

ILRS_ID = '7603901'  # lageos 1
# ILRS_ID = '9207002'  # lageos 2

PROVIDER = 'SGF'

def main():
    data = slr.SlrDlManager(USER, PASSWD)

    cpf_list = data.queryCpfData(ILRS_ID, REF_TIME)
    print(cpf_list)

    cpf = cpf_list[cpf_list['provider'] == PROVIDER]
    start = datetime.datetime.strptime(cpf.start_data_date.item(), '%Y-%m-%d %H:%M:%S')
    stop = datetime.datetime.strptime(cpf.end_data_date.item() , '%Y-%m-%d %H:%M:%S')
    idx = [cpf.index.item(), ]

    cpf = data.dlAndParseCpfData(idx, start, stop)
    print(cpf)


if __name__ == "__main__":
    main()

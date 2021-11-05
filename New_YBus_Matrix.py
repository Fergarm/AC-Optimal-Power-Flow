import numbers
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def Make_New_Ybus (baseMVA, Bus_Data,Lines_Data):

    Columns_Name_Buses = Bus_Data.columns
    Columns_Name_Lines = Lines_Data.columns

    nb = len(Bus_Data)
    nl = len(Lines_Data)

    BS_Matrix = pd.DataFrame(np.zeros([nb,nb]))
    Smax_Matrix = pd.DataFrame(np.zeros([nb,nb]))

    for r in range(len(Lines_Data)):
        Smax_Matrix.loc[Lines_Data['fbus'][r],Lines_Data['tbus'][r]] = Lines_Data['Smax'][r]
        Smax_Matrix.loc[Lines_Data['tbus'][r],Lines_Data['fbus'][r]] = Lines_Data['Smax'][r]
        BS_Matrix.loc[Lines_Data['fbus'][r],Lines_Data['tbus'][r]] = Lines_Data['b'][r]
        BS_Matrix.loc[Lines_Data['tbus'][r],Lines_Data['fbus'][r]] = Lines_Data['b'][r]

    Bus_Data = Bus_Data.to_numpy()
    Lines_Data = Lines_Data.to_numpy()

    Gs = np.where(Columns_Name_Buses == 'Gs')[0][0]
    Bs = np.where(Columns_Name_Buses == 'Bs')[0][0]
    status = np.where(Columns_Name_Lines == 'status')[0][0]
    R = np.where(Columns_Name_Lines == 'r')[0][0]
    X = np.where(Columns_Name_Lines == 'x')[0][0]
    B = np.where(Columns_Name_Lines == 'b')[0][0]
    Tap = np.where(Columns_Name_Lines == 'ratio')[0][0]
    Shift = np.where(Columns_Name_Lines == 'angle')[0][0]
    fbus = np.where(Columns_Name_Lines == 'fbus')[0][0]
    tbus = np.where(Columns_Name_Lines == 'tbus')[0][0]


    stat = Lines_Data[:, status]                                    ## ones at in-service branches
    Ys = stat / (Lines_Data[:, R] + 1j * Lines_Data[:, X])          ## series admittance
    Bc = stat * Lines_Data[:, B]                                    ## line charging susceptance
    tap = np.ones(nl)                                               ## default tap ratio = 1
    i = np.nonzero(Lines_Data[:, Tap])                              ## indices of non-zero tap ratios
    tap[i] = Lines_Data[i, Tap]                                     ## assign non-zero tap ratios
    tap = tap * np.exp(1j * np.pi / 180 * Lines_Data[:, Shift])     ## add phase shifters

    Ytt = Ys + 1j * Bc / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = - Ys / np.conj(tap)
    Ytf = - Ys / tap

    Ysh = (Bus_Data[:, Gs] + 1j * Bus_Data[:, Bs]) / baseMVA

    f = Lines_Data[:, fbus]
    t = Lines_Data[:, tbus]

    Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
    Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb)) 

    i = np.r_[range(nl), range(nl)]

    Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])), (nl, nb))
    Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])), (nl, nb))

    Ybus = Cf.T * Yf + Ct.T * Yt + \
        csr_matrix((Ysh, (range(nb), range(nb))), (nb, nb))

    YBus = pd.DataFrame(Ybus.todense())

    return YBus,  Smax_Matrix, BS_Matrix




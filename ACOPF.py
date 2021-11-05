import numpy as np
import pandas as pd
from pyomo.environ import * 
from YBus_Matrix import Make_New_Ybus
import pyomo.environ as pyo

######### NETWORK PARAMETERS ##########
File = pd.ExcelFile('Data_14_Bus.xlsx')

Data_Ybus = pd.read_excel(File, sheet_name='input_Ybus')
Lines = pd.read_excel(File, sheet_name='Adj_Matrix')
Bus_Data = pd.read_excel(File, sheet_name='Buses')
Costs_Data = pd.read_excel(File, sheet_name='Costs')

NBus = len(Bus_Data)
refB = int(Bus_Data['refB'][0])

sb = 100

YBus, Smax, BS = Make_New_Ybus(sb,Bus_Data,Data_Ybus)

model = pyo.AbstractModel()

## -- SETs -- ##
model.B = pyo.Set(initialize=np.arange(1,NBus+1))
model.refB = pyo.Set(initialize=np.arange(refB,refB+1))

def Set_L_init(model):
    return ((i+1,j+1) for i in np.arange(0,NBus) for j in np.arange(0,NBus) if Lines.loc[i,j]==1)
model.L = pyo.Set(dimen=2,initialize=Set_L_init)
    
## -- PARAMETERS -- ##
model.PGmin = pyo.Param(model.B, initialize = 0)

def Ca_init(model,i):
    return Costs_Data['a'].loc[i-1]
model.Ca = pyo.Param(model.B, rule=Ca_init)

def Cb_init(model,i):
    return Costs_Data['b'].loc[i-1]
model.Cb = pyo.Param(model.B, rule=Cb_init)

def Cc_init(model,i):
    return Costs_Data['c'].loc[i-1]
model.Cc = pyo.Param(model.B, rule=Cc_init)

def PL_init(model,i):
    return Bus_Data['PL'].loc[i-1]
model.PL = pyo.Param(model.B, rule=PL_init)

def QL_init(model,i):
    return Bus_Data['QL'].loc[i-1]
model.QL = pyo.Param(model.B, rule=QL_init)

def PGmax_init(model,i):
    return Bus_Data['PGmax'].loc[i-1]
model.PGmax = pyo.Param(model.B, rule=PGmax_init)

def QGmin_init(model,i):
    return Bus_Data['QGmin'].loc[i-1]
model.QGmin = pyo.Param(model.B,rule=QGmin_init)

def QGmax_init(model,i):
    return Bus_Data['QGmax'].loc[i-1]
model.QGmax = pyo.Param(model.B, rule=QGmax_init)

def Vmax_init(model,i):
    return Bus_Data['Vmax'].loc[i-1]
model.Vmax = pyo.Param(model.B, rule=Vmax_init)

def Vmin_init(model,i):
    return Bus_Data['Vmin'].loc[i-1]
model.Vmin = pyo.Param(model.B, rule=Vmin_init)

def BS_init(model,i,j):
    return BS.loc[i-1,j-1]
model.bs = pyo.Param(model.B,model.B, rule=BS_init)

def Smax_init(model,i,j):
    return Smax.loc[i-1,j-1]
model.s_max = pyo.Param(model.B,model.B,rule=Smax_init)

def Gij_init(model,i,j):
    if i != j:
        return round(YBus.loc[i-1,j-1].real,4)
    else:
        return 0
model.Gij = pyo.Param(model.B,model.B, within = pyo.Reals, rule=Gij_init)

def Bij_init(model,i,j):
    if i != j:
        return round(YBus.loc[i-1,j-1].imag,4)
    else:
        return 0
model.Bij = pyo.Param(model.B,model.B, within = pyo.Reals, rule=Bij_init)

def Gi_init (model,i):
    return round(YBus.loc[i-1,i-1].real,2)
model.Gi = pyo.Param(model.B, within = pyo.Reals, rule=Gi_init)

def Bi_init (model,i):
    return round(YBus.loc[i-1,i-1].imag,2)
model.Bi = pyo.Param(model.B, within = pyo.Reals, rule=Bi_init)

model.sb = pyo.Param(initialize = sb)

## -- VARIABLES -- ##
model.pg = pyo.Var(model.B, within = pyo.NonNegativeReals, initialize = 0)
model.qg = pyo.Var(model.B,initialize = 0)
model.V = pyo.Var(model.B, initialize = 0)
model.Th = pyo.Var(model.B, bounds = (-8*atan(1),8*atan(1)))
model.P = pyo.Var(model.L, initialize = 1)
model.Q = pyo.Var(model.L, initialize = 0)

## -- OBJECTIVE FUNCTION -- ##
def Fun_obj(model):
    return sum(model.Ca[i]*(model.pg[i]*100)**2 + model.Cb[i]*(model.pg[i]*100) + model.Cc[i] for i in model.B)
model.FunObj = pyo.Objective(rule = Fun_obj, sense = pyo.minimize)

## -- CONSTRAINTS -- ##
def NPFE_R (model,i):
    value1 = model.V[i]**2*model.Gi[i] + model.V[i]*sum(model.V[l]*(model.Gij[r,l]*cos(model.Th[r]-model.Th[l]) + model.Bij[r,l]*sin(model.Th[r]-model.Th[l])) for (r,l) in model.L if r==i)
    return model.pg[i] - model.PL[i]/100 == value1
model.npfe_r = pyo.Constraint(model.B, rule=NPFE_R)

def NPFE_Im (model,i):
    value2 = (-1)*model.V[i]**2*model.Bi[i] + model.V[i]*sum(model.V[l]*(model.Gij[r,l]*sin(model.Th[r]-model.Th[l]) - model.Bij[r,l]*cos(model.Th[r]-model.Th[l])) for (r,l) in model.L if r==i)
    return model.qg[i] - model.QL[i]/100 == value2
model.npfe_im = pyo.Constraint(model.B, rule=NPFE_Im)
        
def LPFE_R (model, i,j):
    value_3 = model.V[i]*model.V[j]*(model.Gij[i,j]*cos(model.Th[i]- model.Th[j]) + model.Bij[i,j]*sin(model.Th[i]- model.Th[j])) - model.Gij[i,j]*model.V[i]**2
    return model.P[i,j] == value_3
model.lpfe_r = pyo.Constraint(model.L, rule=LPFE_R)
    
def LPFE_Im (model, i,j):
    value_4 = model.V[i]*model.V[j]*(model.Gij[i,j]*sin(model.Th[i]- model.Th[j]) - model.Bij[i,j]*cos(model.Th[i]- model.Th[j])) + model.V[i]**2*(model.Bij[i,j] - model.bs[i,j])
    return model.Q[i,j] == value_4
model.lpfe_im = pyo.Constraint(model.L, rule=LPFE_Im)

def GL_PG (model, i):
    return model.pg[i] <= model.PGmax[i]/100
model.gl_pg = pyo.Constraint(model.B, rule=GL_PG)

def GL_QG (model, i):
    return inequality(model.QGmin[i]/100, model.qg[i],model.QGmax[i]/100)
model.gl_qg = pyo.Constraint(model.B, rule=GL_QG)

def LSL (model,i,j):
    return sqrt(model.P[i,j]**2 + model.Q[i,j]**2) <= model.s_max[i,j]/100
model.lsl = pyo.Constraint(model.L, rule=LSL)

def VL (model,i):
    return inequality(model.Vmin[i], model.V[i], model.Vmax[i])
model.vl = pyo.Constraint(model.B, rule=VL)

def SUB_B (model, sub):
    return model.Th[sub] == 0.0
model.sub_bus = pyo.Constraint(model.refB, rule=SUB_B)

instance = model.create_instance()
opt = SolverFactory("bonmin")
results = opt.solve(instance)
results.write()

OF_value = round(value(instance.FunObj),4)
print(OF_value)

## -- FILE RESULTS -- ##

Flow_lines_table = pd.DataFrame(np.zeros([len(instance.L),6]), columns=['from','to','Pij', 'Qij','Flow_S','Smax'])
Power_table = pd.DataFrame(np.zeros([len(instance.B),5]), columns=['bus','PG', 'QG','PL','QL'])
Data_table = pd.DataFrame(np.zeros([len(instance.B),5]), columns=['bus','Voltage', 'Angle','Gi','Bi']) 
Ymatrix_table = pd.DataFrame(np.zeros([len(instance.L),5]), columns=['from','to','Gij', 'Bij','BS'])

r=0
for i in instance.B:
    Power_table.loc[r,'bus'] = str(i)
    Power_table.loc[r,'PG'] = round(instance.pg[i].value,4)*100
    Power_table.loc[r,'QG'] = round(instance.qg[i].value,4)*100
    Power_table.loc[r,'PL'] = round(instance.PL[i],4)
    Power_table.loc[r,'QL'] = round(instance.QL[i],4)
    r += 1

r=0
for i in instance.B:
    Data_table.loc[r,'bus'] = str(i)
    Data_table.loc[r,'Voltage'] = round(instance.V[i].value,4)
    Data_table.loc[r,'Angle'] = round(instance.Th[i].value,4)
    Data_table.loc[r,'Gi'] = instance.Gi[i]
    Data_table.loc[r,'Bi'] = instance.Bi[i]
    r += 1

r=0
for (i,j) in instance.L:
    Flow_lines_table.loc[r,'from'] = str(i)
    Flow_lines_table.loc[r,'to'] = str(j)
    Flow_lines_table.loc[r,'Pij'] = round(instance.P[i,j].value,4)*100
    Flow_lines_table.loc[r,'Qij'] = round(instance.Q[i,j].value,4)*100
    Flow_lines_table.loc[r,'Flow_S'] = round(sqrt((instance.P[i,j].value)**2+(instance.Q[i,j].value)**2)*100,4)
    Flow_lines_table.loc[r,'Smax'] = instance.s_max[i,j]

    Ymatrix_table.loc[r,'from'] = str(i)
    Ymatrix_table.loc[r,'to'] = str(j)
    Ymatrix_table.loc[r,'Gij'] = instance.Gij[i,j]
    Ymatrix_table.loc[r,'Bij'] = instance.Bij[i,j]
    Ymatrix_table.loc[r,'BS'] = instance.bs[i,j]
    r += 1  

with pd.ExcelWriter('ACOPF_Results.xlsx') as data: 
    Power_table.to_excel(data, sheet_name="Power_injected")
    Data_table.to_excel(data, sheet_name="Voltage_Angle")
    Flow_lines_table.to_excel(data, sheet_name="Lines_Flow")
    Ymatrix_table.to_excel(data,sheet_name='Ymatrix')

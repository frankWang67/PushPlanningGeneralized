#####################
#    CA3P & RRTc    #
#####################

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

item_name_list = ['Length', 'Time', 'Success', 'Nodenum']

method = ['CA3P','RRTc']

success_data = pd.read_csv('./Success.csv')

for idx, item_name in enumerate(item_name_list):
    exp_data = pd.read_csv('./'+item_name+'.csv')
    exp_data['Method'] = exp_data.Method.map(lambda x: 'CA3P' if x =='R3T' else 'RRTc')

    data_for_CA3P = []
    data_for_RRTc = []

    if item_name == "Success":
        for scene_idx in range(5):
            CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_CA3P.append(len(CA3P_data))

            RRTc_data = exp_data[(exp_data['Method']=="RRTc") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_RRTc.append(len(RRTc_data))
    elif item_name == "Nodenum":
        for scene_idx in range(5):
            CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_CA3P.append((CA3P_data["Nodenum"].mean(), CA3P_data["Nodenum"].std()))

            RRTc_data = exp_data[(exp_data['Method']=="RRTc") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_RRTc.append((RRTc_data["Nodenum"].mean(), RRTc_data["Nodenum"].std()))
    elif item_name == "Length":
        for scene_idx in range(5):
            CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_CA3P.append((CA3P_data["Length"].mean(), CA3P_data["Length"].std()))

            RRTc_data = exp_data[(exp_data['Method']=="RRTc") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_RRTc.append((RRTc_data["Length"].mean(), RRTc_data["Length"].std()))
    elif item_name == "Time":
        for scene_idx in range(5):
            CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_CA3P.append((CA3P_data["Time"].mean(), CA3P_data["Time"].std()))

            RRTc_data = exp_data[(exp_data['Method']=="RRTc") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
            data_for_RRTc.append((RRTc_data["Time"].mean(), RRTc_data["Time"].std()))
    
    print(item_name)
    print("--------------------")
    print("CA3P")
    print(data_for_CA3P)
    print("RRTc")
    print(data_for_RRTc)



##########################
#    CA3P & CA3P(old)    #
##########################

# import seaborn as sns
# from matplotlib import pyplot as plt
# import pandas as pd
# from matplotlib.font_manager import FontProperties
# from matplotlib import font_manager
# import os
# import sys
# DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# item_name_list = ['Length', 'Time', 'Success', 'Nodenum']

# method = ['CA3P','CA3P_old']

# success_data = pd.read_csv('./Success.csv')

# for idx, item_name in enumerate(item_name_list):
#     exp_data = pd.read_csv('./'+item_name+'.csv')
#     exp_data['Method'] = exp_data.Method.map(lambda x: 'CA3P' if x =='R3T' else 'CA3P_old')

#     data_for_CA3P = []
#     data_for_CA3P_old = []

#     if item_name == "Success":
#         for scene_idx in range(5):
#             CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P.append(len(CA3P_data))

#             RRTc_data = exp_data[(exp_data['Method']=="CA3P_old") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P_old.append(len(RRTc_data))
#     elif item_name == "Nodenum":
#         for scene_idx in range(5):
#             CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P.append((CA3P_data["Nodenum"].mean(), CA3P_data["Nodenum"].std()))

#             RRTc_data = exp_data[(exp_data['Method']=="CA3P_old") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P_old.append((RRTc_data["Nodenum"].mean(), RRTc_data["Nodenum"].std()))
#     elif item_name == "Length":
#         for scene_idx in range(5):
#             CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P.append((CA3P_data["Length"].mean(), CA3P_data["Length"].std()))

#             RRTc_data = exp_data[(exp_data['Method']=="CA3P_old") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P_old.append((RRTc_data["Length"].mean(), RRTc_data["Length"].std()))
#     elif item_name == "Time":
#         for scene_idx in range(5):
#             CA3P_data = exp_data[(exp_data['Method']=="CA3P") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P.append((CA3P_data["Time"].mean(), CA3P_data["Time"].std()))

#             RRTc_data = exp_data[(exp_data['Method']=="CA3P_old") & (exp_data['Scene']==scene_idx) & (success_data['success']==True)]
#             data_for_CA3P_old.append((RRTc_data["Time"].mean(), RRTc_data["Time"].std()))
    
#     print(item_name)
#     print("--------------------")
#     print("CA3P")
#     print(data_for_CA3P)
#     print("CA3P_old")
#     print(data_for_CA3P_old)

import os
import numpy as np
import pandas as pd
import pickle

FAILURE_EXECUTION_TIME = 130.0
FAILURE_PATH_LENGTH = 2.0

def convert2csv():
    scene_list = [0, 1, 2, 3, 4]
    r3t_round_list_old = [1, 2, 3]
    r3t_round_list = [1, 2, 4]
    rrt_round_list = [0, 1, 2]

    Nodenum_data = {'Method': [],
                    'Scene': [],
                    'Nodenum': [],
                    'Samplenum': []}
    
    Time_data = {'Method': [],
                 'Scene': [],
                 'Time': []}
    
    Dist2goal_data = {'Method': [],
                      'Scene': [],
                      'Dist2goal': []}
    
    Success_data = {'Method': [],
                    'Scene': [],
                    'success': []}
    
    Length_data = {'Method': [],
                   'Scene': [],
                   'Length': []}
    
    # R3T data
    R3T_path = '/home/yongpeng/research/R3T_shared/data/exp/new_model/all_in_one'
    for scene_idx in scene_list:
        for round_idx in r3t_round_list:
            pkl_file_path = os.path.join(R3T_path, \
                                         'scene_{0}_10_round_#{1}'.format(scene_idx, round_idx), \
                                         'report.pkl')
            data = pickle.load(open(pkl_file_path, 'rb'))
            num_samples = len(data['node_num'])

            Nodenum_data['Method'] += ['R3T' for i in range(num_samples)]
            Nodenum_data['Scene'] += [scene_idx for i in range(num_samples)]
            Nodenum_data['Nodenum'] += data['node_num']
            Nodenum_data['Samplenum'] += data['node_num']

            Time_data['Method'] += ['R3T' for i in range(num_samples)]
            Time_data['Scene'] += [scene_idx for i in range(num_samples)]

            planning_time_data = data['planning_time']
            for i in range(len(planning_time_data)):
                if planning_time_data[i] >= FAILURE_EXECUTION_TIME or data['success'][i] == False:
                    planning_time_data[i] = FAILURE_EXECUTION_TIME
            Time_data['Time'] += planning_time_data

            Dist2goal_data['Method'] += ['R3T' for i in range(num_samples)]
            Dist2goal_data['Scene'] += [scene_idx for i in range(num_samples)]
            Dist2goal_data['Dist2goal'] += data['distance_to_goal']

            Success_data['Method'] += ['R3T' for i in range(num_samples)]
            Success_data['Scene'] += [scene_idx for i in range(num_samples)]
            Success_data['success'] += data['success']

            Length_data['Method'] += ['R3T' for i in range(num_samples)]
            Length_data['Scene'] += [scene_idx for i in range(num_samples)]

            path_length_data = np.minimum(data['path_length'], FAILURE_PATH_LENGTH).tolist()
            for i in range(len(path_length_data)):
                if path_length_data[i] == 0 or data['success'][i] == False:
                    path_length_data[i] = 2.0
            Length_data['Length'] += path_length_data

    # R3T data (old model)
    R3T_path = '/home/yongpeng/research/R3T_shared/data/exp'
    for scene_idx in scene_list:
        for round_idx in r3t_round_list_old:
            pkl_file_path = os.path.join(R3T_path, \
                                         'scene_{0}_10_round_#{1}'.format(scene_idx, round_idx), \
                                         'report.pkl')
            data = pickle.load(open(pkl_file_path, 'rb'))
            num_samples = len(data['node_num'])

            Nodenum_data['Method'] += ['R3T_old' for i in range(num_samples)]
            Nodenum_data['Scene'] += [scene_idx for i in range(num_samples)]
            Nodenum_data['Nodenum'] += data['node_num']
            Nodenum_data['Samplenum'] += data['node_num']

            Time_data['Method'] += ['R3T_old' for i in range(num_samples)]
            Time_data['Scene'] += [scene_idx for i in range(num_samples)]

            planning_time_data = data['planning_time']
            for i in range(len(planning_time_data)):
                if planning_time_data[i] >= FAILURE_EXECUTION_TIME or data['success'][i] == False:
                    planning_time_data[i] = FAILURE_EXECUTION_TIME
            Time_data['Time'] += planning_time_data

            Dist2goal_data['Method'] += ['R3T_old' for i in range(num_samples)]
            Dist2goal_data['Scene'] += [scene_idx for i in range(num_samples)]
            Dist2goal_data['Dist2goal'] += data['distance_to_goal']

            Success_data['Method'] += ['R3T_old' for i in range(num_samples)]
            Success_data['Scene'] += [scene_idx for i in range(num_samples)]
            Success_data['success'] += data['success']

            Length_data['Method'] += ['R3T_old' for i in range(num_samples)]
            Length_data['Scene'] += [scene_idx for i in range(num_samples)]

            path_length_data = np.minimum(data['path_length'], FAILURE_PATH_LENGTH).tolist()
            for i in range(len(path_length_data)):
                if path_length_data[i] == 0 or data['success'][i] == False:
                    path_length_data[i] = 2.0
            Length_data['Length'] += path_length_data

    # # RRTc data
    # RRTc_path = '/home/yongpeng/research/rrt-algorithms/data/exp/search_space_0.0_0.5_0.0_0.5'
    # for scene_idx in scene_list:
    #     for round_idx in rrt_round_list:
    #         pkl_file_path = os.path.join(RRTc_path, \
    #                                      'scene_{0}_10_round_#{1}'.format(scene_idx, round_idx), \
    #                                      'report.pkl')
    #         data = pickle.load(open(pkl_file_path, 'rb'))
    #         num_samples = len(data['node_num'])

    #         Nodenum_data['Method'] += ['RRTc' for i in range(num_samples)]
    #         Nodenum_data['Scene'] += [scene_idx for i in range(num_samples)]
    #         Nodenum_data['Nodenum'] += data['node_num']
    #         Nodenum_data['Samplenum'] += data['sample_num']

    #         Time_data['Method'] += ['RRTc' for i in range(num_samples)]
    #         Time_data['Scene'] += [scene_idx for i in range(num_samples)]

    #         planning_time_data = data['planning_time']
    #         for i in range(len(planning_time_data)):
    #             if planning_time_data[i] >= FAILURE_EXECUTION_TIME or data['success'][i] == False:
    #                 planning_time_data[i] = FAILURE_EXECUTION_TIME
    #         Time_data['Time'] += planning_time_data

    #         Dist2goal_data['Method'] += ['RRTc' for i in range(num_samples)]
    #         Dist2goal_data['Scene'] += [scene_idx for i in range(num_samples)]
    #         Dist2goal_data['Dist2goal'] += data['distance_to_goal']

    #         Success_data['Method'] += ['RRTc' for i in range(num_samples)]
    #         Success_data['Scene'] += [scene_idx for i in range(num_samples)]
    #         Success_data['success'] += data['success']

    #         Length_data['Method'] += ['RRTc' for i in range(num_samples)]
    #         Length_data['Scene'] += [scene_idx for i in range(num_samples)]
            
    #         path_length_data = np.minimum(data['path_length'], FAILURE_PATH_LENGTH).tolist()
    #         for i in range(len(path_length_data)):
    #             if path_length_data[i] == 0 or data['success'][i] == False:
    #                 path_length_data[i] = 2.0
    #         Length_data['Length'] += path_length_data

    # dataframe
    Nodenum_pd = pd.DataFrame(Nodenum_data)
    Time_pd = pd.DataFrame(Time_data)
    Dist2goal_pd = pd.DataFrame(Dist2goal_data)
    Success_pd = pd.DataFrame(Success_data)
    Length_pd = pd.DataFrame(Length_data)

    Nodenum_pd.to_csv('./Nodenum.csv')
    Time_pd.to_csv('./Time.csv')
    Dist2goal_pd.to_csv('./Dist2goal.csv')
    Success_pd.to_csv('./Success.csv')
    Length_pd.to_csv('./Length.csv')


if __name__ == '__main__':
    convert2csv()

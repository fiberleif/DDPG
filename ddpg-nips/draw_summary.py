from visualize import Visualizer
import os
import numpy as np

YOUR_ALIAS = 'youralias'
__color_cnt = 0
color_list = [('rgb(0,100,80)', 'rgba(0,100,80,0.2)'), ('rgb(0,176,246)', 'rgba(0,176,246,0.2)'), ('rgb(231,107,243)', 'rgba(231,107,243,0.2)'), ('rgb(255,204,0)', 'rgba(255,204,0,0.2)')]

def get_next_color():
    global __color_cnt, color_list
    ret = color_list[__color_cnt]
    __color_cnt += 1
    if __color_cnt == len(color_list):
        __color_cnt = 0
    return ret

def update_from_file(f, run_name):
    steps, train_rewards, eval_rewards = [], [], []
    for line in f:
        s = line.split('|')
        if 'total/steps' in line:
            steps.append(float(s[2])/1e6)
        elif 'epoch/reward_sum' in line:
            train_rewards.append(float(s[2]))
        elif 'eval/reward_sum' in line:
            eval_rewards.append(float(s[2]))
    return [steps, train_rewards, eval_rewards]

def overall_data_info(data, axis):
    steps = []
    for item in data:
        if len(item[0]) > len(steps):
            steps = item[0]
    v_ave, v_upper, v_lower = [], [], []
    for i, step in enumerate(steps):
        v = []
        for item in data:
            if len(item[axis]) > i:
                v.append(item[axis][i])
        if len(v) > 0:
            v_ave.append(np.mean(v))
            v_upper.append(np.mean(v) + np.std(v))
            v_lower.append(np.mean(v) + np.std(v))
    return steps, v_ave, v_upper, v_lower

def draw_datas(visualizer, datas):
    for name, data in datas.items():
        ## train_rewards
        steps, v_ave, v_upper, v_lower = overall_data_info(data, 1)
        color, color_a = get_next_color()
        visualizer.draw_line(name+"_train_rewards", color, steps, v_ave)
        visualizer.fill_line(name+"_train_rewards"+"_fill", color_a, steps, v_upper, v_lower)
        ## eval_rewards
        steps, v_ave, v_upper, v_lower = overall_data_info(data, 2)
        color, color_a = get_next_color()
        visualizer.draw_line(name+"_eval_rewards", color, steps, v_ave)
        visualizer.fill_line(name+"_eval_rewards"+"_fill", color_a, steps, v_upper, v_lower)

def summary(visualizer):
    datas = {}
    cwd = os.path.join(os.getcwd(), 'log')
    subdirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    for dname in subdirs:
        dlist = dname.split('-')
        if len(dlist) != 8:
            print(dname, 'seems not to be a log directory.')
            continue
        run_name = dlist[0]
        if not run_name in datas:
            datas[run_name] = []
        f = open(os.path.join(os.path.join(cwd, dname), 'log.txt'), 'r')
        datas[run_name].append(update_from_file(f, run_name))
    draw_datas(visualizer, datas)


visualizer = Visualizer(YOUR_ALIAS, 'summary')
summary(visualizer)
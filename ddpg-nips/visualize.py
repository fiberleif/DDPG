##################################
##  Author: Tianyang Zhang
##  Contact: keavilzhangzty@gmail.com

from visdom import Visdom
from baselines import logger

class Visualizer(object):
    def __init__(self, envid, run_name, xaxis_name='steps x 1e6', yaxis_name='return', server='http://168.62.43.74', port=5000):
        self.viz = Visdom(server=server, port=port)
        assert self.viz.check_connection()  
        self.run_name = run_name
        self.traces = {}
        self.win = None
        self.envid = envid
        self.layout = dict(
            title=run_name, 
            xaxis={'title': xaxis_name}, 
            yaxis={'title': yaxis_name}
            )   

    def initialize(self, name, color):
        self.traces[name] = dict(
            x=[0], y=[0], name=name, 
            line=dict(color=color, width=3),
            mode="lines", type='custom'
            )

    def send(self):
        try:
            logger.info('Send graph to server.')
            self.win=self.viz._send({
                'data':list(self.traces.values()), 
                'layout':self.layout, 
                'win':self.win,
                'eid':self.envid,
                })
        except:
            logger.info('Error: Send graph error! This error will be ignored.')
            self.win = None

    def paint(self, name, steps, y1):
        self.traces[name]['x'].append(steps)
        self.traces[name]['y'].append(y1)
        self.send()

    def draw_line(self, name, color, x, y):
        self.traces[name] = dict(
            x=x, y=y, name=name, 
            line=dict(color=color, width=3),
            mode="lines", type='custom'
            )
        self.send()

    def fill_line(self, name, color, x, y_upper, y_lower):
        self.traces[name] = dict(
            x=x+x[::-1], 
            y=y_upper+y_lower[::-1],
            name=name, 
            line=dict(color='rgba(255,255,255,0)'),
            fill='tozerox',
            fillcolor=color,
            showlegend=False,
            type='custom'
            )
        self.send()

#
if __name__ == '__main__':
    v = Visualizer('youralias' ,'ddpg')
    v.draw_line('sth', 'rgb(0, 100, 80)', [0,1,2,3,4], [0, 5, 7, 1])
    #v.fill_line('sth_fill', 'rgba(0, 100, 80, 0.2)', [0,1,2,3,4], [0,6,8,2,4], [0,4,6,0,2])
    exit()
    v.initialize('train-reward', 'red')
    v.paint('train-reward', 1000000, 1)
    v.initialize('test-reward', 'cyan')
    v.paint('test-reward', 1000000, 3)
    input()
    v.paint('test-reward', 2000000, 1)
    v.paint('train-reward', 2000000, 2)
    input()
    v.paint('test-reward', 3000000, 1)
    v.paint('train-reward', 3000000, 2)
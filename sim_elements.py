import numpy as np
import networkx as nx
import warnings
import os
from networkx.algorithms.shortest_paths.weighted import dijkstra_path

from utils import *

_sim_info = {}

def push_to_sim_info(key, value):
    global _sim_info
    if key in _sim_info.keys():
        warnings.warn("Key already exists! Replacing its current value...")
    _sim_info[key] = value

def pull_from_sim_info(key):
    return _sim_info[key]

class Clock:
    def __init__(self, max_time, start_time=0, interval=1, mode='forward'):
        self._max_time = max_time
        self._start_time = self._time = start_time
        self._interval = interval
        self._mode = mode
        
    def get_time(self):
        return self._time
    
    def tick(self):
        self._time += self._interval
        
    def switch_mode(self):
        if self._mode == 'forward':
            self._time = self._max_time
            self._mode = 'backward'
            self._interval *= -1
        else:
            self._time = self._start_time
            self._mode = 'forward'
            self._interval *= -1
    
    def reset(self):
        self._time = self._start_time
        if self._mode == 'backward':
            self._interval = 1
        self._mode = 'forward'
        
class Controller:
    def __init__(self, link_info, T, Tm, models, demands, cost_params, dtchoices=None, srates=None):
        self.link_info = link_info
        self.T = T
        self.Tm = Tm
        self.models = models
        self.demands = demands
        self.cost_params = cost_params
        self.dtchoices = dtchoices
        self.srates = srates
        
        tmpgraph = nx.DiGraph(link_info)
        self.sources, self.sinks, self.diverges, self.merges = get_node_types(tmpgraph)
        
        self.ntypes = len(self.sinks)
        self.init_choices(tmpgraph)
        
        for nodeid in self.sources:
            tmpgraph.node[nodeid]['nodeid'] = nodeid
            tmpgraph.node[nodeid]['nodetype'] = 'source'
            tmpgraph.node[nodeid]['demands'] = demands[nodeid]
            tmpgraph.node[nodeid]['dtchoices'] = self.dtchoices[nodeid]
        for iterid,nodeid in enumerate(self.sinks):
            tmpgraph.node[nodeid]['nodeid'] = nodeid
            tmpgraph.node[nodeid]['nodetype'] = 'sink'
            tmpgraph.node[nodeid]['sinkno'] = iterid
        for nodeid in self.diverges:
            tmpgraph.node[nodeid]['nodeid'] = nodeid
            tmpgraph.node[nodeid]['nodetype'] = 'diverge'
            tmpgraph.node[nodeid]['srates'] = self.srates[nodeid]
        for nodeid in self.merges:
            tmpgraph.node[nodeid]['nodeid'] = nodeid
            tmpgraph.node[nodeid]['nodetype'] = 'merge'
        
        push_to_sim_info('T', T)
        push_to_sim_info('ntypes', self.ntypes)
        for key,value in cost_params.items():
            push_to_sim_info(key,value)
        
        self.network = Network(tmpgraph, models)
    
    def init_choices(self, init_graph):
        paths = {snknode:[dijkstra_path(init_graph, srcnode, snknode, weight='length') for srcnode in self.sources] for snknode in self.sinks}
        def dtchoice_init():
            choices = np.zeros((self.T, self.ntypes))
            choices[np.random.choice(self.Tm, 4),:] = 0.25
            return choices
        
        def srate_init(nodeid):
            out_degree = init_graph.out_degree(nodeid)
            choices = np.zeros((self.T, out_degree, self.ntypes))
            for i in np.arange(self.ntypes):
                for j,srcnodeid in enumerate(self.sources):
                    if nodeid in paths[self.sinks[i]][j]:
                        nxtnodeid = paths[self.sinks[i]][j][paths[self.sinks[i]][j].index(nodeid)+1]
                        try:
                            idx = list(init_graph.out_edges(nodeid)).index((nodeid, nxtnodeid))
                            choices[:,idx,i] = 1
                        except ValueError:
                            pass
                        break
                if np.any(~np.isclose(1, np.sum(choices[:,:,i], axis=1))):
                    choices[:,np.random.choice(out_degree, 1),i] = 1
            return choices

        if self.dtchoices == None:
            self.dtchoices = dict()
            for nodeid in self.sources:
                self.dtchoices[nodeid] = dtchoice_init()
        if self.srates == None:
            self.srates = dict()
            for nodeid in self.diverges:
                self.srates[nodeid] = srate_init(nodeid)
                
    def calc_DUE(self, niter, learning_rate, out_file='out.csv', print_iter_measures=False):
        with open(out_file, 'w') as f:
            for iterno in np.arange(niter):
                self.network.runsim()
                self.network.compute_costs()
                dtchoices_old = self.dtchoices
                srates_old = self.srates

                dtccosts = dict()
                srcosts = dict()
                for node in self.network.nodes:
                    if node.nodetype == 'source':
                        dtccosts[node.nodeid] = node.get_outlink_costs().squeeze() # since only one outlink
                    elif node.nodetype == 'diverge':
                        srcosts[node.nodeid] = node.get_outlink_costs()

                dtchoices_new = dict()
                srates_new = dict()

                # update dtchoices
                for sourcenodeid in self.sources:
                    dtchoices_new[sourcenodeid] = np.zeros_like(self.dtchoices[sourcenodeid])
                    for iterid,sinknodeid in enumerate(self.sinks):
                        tmp = solveqp(self.dtchoices[sourcenodeid][:self.Tm, iterid], dtccosts[sourcenodeid][:self.Tm, iterid], lr=learning_rate)['x']
                        tmp[tmp<MIN_PROP_VALUE] = 0
                        dtchoices_new[sourcenodeid][:self.Tm, iterid] = tmp/np.sum(tmp)
                        assert np.isclose(np.sum(dtchoices_new[sourcenodeid][:self.Tm, iterid]), 1.)

                # update srates
                for divergenodeid in self.diverges:
                    srates_new[divergenodeid] = np.zeros_like(self.srates[divergenodeid])
                    for iterid,sinknodeid in enumerate(self.sinks):
                        for t in np.arange(self.T):
                            tmp = solveqp(self.srates[divergenodeid][t, :, iterid], srcosts[divergenodeid][t, :, iterid], lr=learning_rate)['x']
                            tmp[tmp<MIN_PROP_VALUE] = 0
                            srates_new[divergenodeid][t, :, iterid] = tmp/np.sum(tmp)
                            assert np.isclose(np.sum(srates_new[divergenodeid][t, :, iterid]), 1.)

                measures = get_measures(dtchoices_new, srates_new, dtchoices_old, srates_old, dtccosts, srcosts)
                f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format(iterno, *measures))
                
                # reset all
                self.reset(dtchoices_new, srates_new)
                if print_iter_measures:
                    print('{0:<5} {1:<14.10} {2:<14.10} {3:<14.10}\t {4:<14.10} {5:<14.10} {6:<14.10}'.format(iterno, *measures))
        
        return self.dtchoices, self.srates
    
    def reset(self, dtchoices_new, srates_new):
        self.dtchoices = dtchoices_new
        self.srates = srates_new
        self.network.reset(dtchoices_new, srates_new)
        
class Network(nx.DiGraph):
    def __init__(self, init_graph, models):
        self.T = pull_from_sim_info('T')
        self.ntypes = pull_from_sim_info('ntypes')
        self.clock = Clock(max_time = self.T-1)
        push_to_sim_info('clock', self.clock)
        
        self.links = []
        for upnodeid, downnodeid, edgedata in init_graph.edges(data=True):
            edgedata['link'] = models['link'](**edgedata)
            self.links.append(edgedata['link'])
        
        nodes = {}
        for nodeid,nodedata in init_graph.nodes(data=True):
            inlinks = [edgedata['link'] for _, _, edgedata in init_graph.in_edges(nodeid, data=True)]
            outlinks = [edgedata['link'] for _, _, edgedata in init_graph.out_edges(nodeid, data=True)]
            nodes[nodeid] = models[nodedata['nodetype']](inlinks, outlinks, **nodedata)
        
        graph_data = []
        for upnodeid, downnodeid, edgedata in init_graph.edges(data=True):
            graph_data.append([nodes[upnodeid], nodes[downnodeid], {'link': edgedata['link']}])
            edgedata['link'].connect_nodes(nodes[upnodeid], nodes[downnodeid])
        
        super().__init__(graph_data)
    
    def runsim(self):
        while self.clock.get_time() < self.T:
            for iterid,node in enumerate(self.nodes):
                node.determine_flows()
            for iterid,link in enumerate(self.links):
                link.update()
            self.clock.tick()
        self.clock.switch_mode()
    
    def compute_costs(self):
        for link in self.links:
            link.init_cost_computation()
        for node in self.nodes:
            node.init_cost_computation()
        
        while self.clock.get_time() >= 0:
            for link in self.links:
                link.compute_costs()
            for node in self.nodes:
                node.compute_costs()
            self.clock.tick()
        self.clock.switch_mode()
    
    def reset(self, dtchoices_new, srates_new):
        self.clock.reset()
        for link in self.links:
            link.reset()
        for node in self.nodes:
            if node.nodetype == 'source':
                node.reset(dtchoices_new[node.nodeid])
            elif node.nodetype == 'diverge':
                node.reset(srates_new[node.nodeid])
            else:
                node.reset()
                
class Link:
    def __init__(self, length, **kwargs):
        self.length = length
        self.T = pull_from_sim_info('T')
        self.ntypes = pull_from_sim_info('ntypes')
        if type(self) == Link:
            self.reset()
    
    def get_time(self):
        return pull_from_sim_info('clock').get_time()
    
    def connect_nodes(self, upnode, downnode):
        self.upnode = upnode
        self.downnode = downnode
    
    def init_cost_computation(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def compute_costs(self):
        raise NotImplementedError
    
    def reset(self):
        self.occs = np.zeros((self.T+1, self.length, self.ntypes))
        self.flows = np.zeros((self.T, self.length+1, self.ntypes))
        
        self.occs_agg = np.zeros((self.T+1, self.length))
        self.flows_agg = np.zeros((self.T, self.length+1))
        
        self.inflow_costs = np.zeros((self.T, self.ntypes))
        
class Node:
    def __init__(self, inlinks, outlinks, nodeid, nodetype):
        self.nodeid = nodeid
        self.inlinks = inlinks
        self.outlinks = outlinks
        self.nodetype = nodetype
        
        self.in_degree = len(self.inlinks)
        self.out_degree = len(self.outlinks)
        
        self.T = pull_from_sim_info('T')
        self.ntypes = pull_from_sim_info('ntypes')
        
        if type(self) == Node:
            self.reset()
    
    def get_time(self):
        return pull_from_sim_info('clock').get_time()
    
    def get_outflows(self, outlink):
        return self.outflows[outlink][self.get_time(),:]
    
    def get_inflows(self, inlink):
        return self.inflows[inlink][self.get_time(),:]
    
    def get_inflow_costs(self, inlink, timesteps):
        return self.inflow_costs[inlink][timesteps,:]
    
    def get_outflow_costs(self, outlink, timestep):
        return self.outflow_costs[outlink][timesteps,:]
    
    def init_cost_computation(self):
        raise NotImplementedError
    
    def determine_flows(self):
        raise NotImplementedError
    
    def compute_costs(self):
        raise NotImplementedError
    
    def get_outlink_costs(self):
        return np.stack([self.outflow_costs[outlink] for outlink in self.outlinks], axis=1)
    
    def reset(self):
        self.inflows = dict()
        self.inflow_costs = dict()
        for inlink in self.inlinks:
            self.inflows[inlink] = np.zeros((self.T, self.ntypes))
            self.inflow_costs[inlink] = np.zeros((self.T, self.ntypes))
        
        self.outflows = dict()
        self.outflow_costs = dict()
        for outlink in self.outlinks:
            self.outflows[outlink] = np.zeros((self.T, self.ntypes))
            self.outflow_costs[outlink] = np.zeros((self.T, self.ntypes))
        
        self.cost_calc_info = dict()

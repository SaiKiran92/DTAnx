from sim_elements import *
from utils import *

ROUND_DECIMALS = 6

class F3Link(Link):
    def __init__(self, length, Q, w, N):
        super().__init__(length)
        self.Q = Q
        self.w = w
        self.N = N
        
        self.alpha = pull_from_sim_info('alpha')
        self.M = pull_from_sim_info('M')
        
        self.reset()
    
    def get_time(self):
        return pull_from_sim_info('clock').get_time()
    
    def connect_nodes(self, upnode, downnode):
        self.upnode = upnode
        self.downnode = downnode
    
    def init_cost_computation(self):
        self.determine_a()
        
    def isSink(self):
        return self.downnode.nodetype == 'sink'
    
    def determine_a(self):
        if self.isSink():
            self.a[:,-1] = self.T
            return
        
        for cellid in np.arange(self.length-self.isSink()):
            self.a[self.A[self.T,cellid]:,cellid] = self.T
            for tau in np.arange(self.A[self.T,cellid]-1):
                self.a[tau,cellid] = np.argmin(self.A[:,cellid] <= tau)-1
    
    def disaggregate(self, cellid, flow_agg):
        cellid %= self.length
        t = self.get_time()
        flows = np.zeros_like(self.flows[0,0,:])
        h = self.A[t,cellid]
        
        if (h < 0):
            return flows
        
        flow_agg_remaining = flow_agg
        while h < t-cellid:
            if h == self.A[t,cellid]:
                iterflow = min(flow_agg_remaining, self.flows_cum[h,0] - self.flows_cum[t-1,cellid+1])
            else:
                iterflow = min(flow_agg_remaining, self.flows_agg[h,0])
            
            flows += iterflow*self.flow_props[h,0,:]
            flow_agg_remaining -= iterflow
            
            if np.isclose(flow_agg_remaining, 0.) and ((not np.isclose(self.flows_cum[h,0], self.flows_cum[t-1,cellid+1]+flow_agg)) or np.isclose(flow_agg, self.Q)):
                break
            h += 1
        
        self.A[t+1,cellid] = h
        flows = np.round(flows, decimals=ROUND_DECIMALS)
        return flows
    
    def update(self):
        t = self.get_time()
        self.flows[t,0,:] = self.upnode.get_outflows(self) # already rounded
        self.flows[t,-1,:] = self.downnode.get_inflows(self)
        self.flows_agg[t,0] = np.sum(self.flows[t,0,:], axis=-1)
        self.flows_agg[t,-1] = np.sum(self.flows[t,-1,:], axis=-1)
        
        midaggflows_unrounded = np.minimum(np.minimum(self.occs_agg[t,:-1], np.maximum(0., self.w*(self.N - self.occs_agg[t,1:]))), self.Q)
        if self.isSink():
            midaggflows_unrounded[-1] = self.occs_agg[t,-2]
        
        for cellid in np.arange(self.length-1):
            self.flows[t,cellid+1,:] = self.disaggregate(cellid, midaggflows_unrounded[cellid])
        
        self.flows_agg[t,1:-1] = np.sum(self.flows[t,1:-1,:], axis=-1)
        self.occs_agg[t+1,:] = self.occs_agg[t,:] + self.flows_agg[t,:-1] - self.flows_agg[t,1:]
        
        self.occs[t+1,...] = self.occs[t,...] + self.flows[t,:-1,:] - self.flows[t,1:,:]
        self.flows_cum[t,:] += self.flows_agg[t,:] + (self.flows_cum[t-1,:] if (t>0) else 0.)
        self.flow_props[t,...] = np.nan_to_num(self.flows[t,...]/self.flows_agg[t,:][:,np.newaxis], nan=0, posinf=0, neginf=0)
    
    def compute_costs(self):
        t = self.get_time()
        
        firsta = self.a[t-1,-1] if (t>0) else (self.length-1 if not self.isSink() else self.T)
        lasta = self.a[t,-1]
        if firsta == self.T:
            if self.isSink() and t < self.T - (self.length-1):
                self.inflow_costs[t,:] = self.alpha*(self.length-1) + self.downnode.get_inflow_costs(self, t+self.length-1)
            else:
                self.inflow_costs[t,:] = self.M*self.alpha
        elif firsta == lasta:
            self.inflow_costs[t,:] = (firsta-t)*self.alpha + self.downnode.get_inflow_costs(self, firsta)
        else:
            # first cohort
            firstprop = np.nan_to_num((self.flows_cum[firsta,-1] - (self.flows_cum[t-1,0] if t>0 else 0))/self.flows_agg[t,0], nan=0, posinf=0, neginf=0)
            self.inflow_costs[t,:] = firstprop*((firsta - t)*self.alpha + self.downnode.get_inflow_costs(self, firsta))
            
            # last cohort
            if not (lasta == self.T):
                lastprop = np.nan_to_num((self.flows_cum[t,0] - self.flows_cum[lasta-1,-1])/self.flows_agg[t,0], nan=1, posinf=1, neginf=1)
                self.inflow_costs[t,:] += lastprop*((lasta - t)*self.alpha + self.downnode.get_inflow_costs(self, lasta))
            else:
                self.inflow_costs[t,:] += np.nan_to_num((self.flows_cum[t,0] - self.flows_cum[-1,-1])/self.flows_agg[t,0], nan=1, posinf=1, neginf=1)*self.M*self.alpha # default == 1, cuz flow could be zero
            
            # mid cohorts
            if lasta - firsta >= 2:
                midas = np.arange(firsta+1, lasta)
                midprops = self.flows_agg[midas,-1]/self.flows_agg[t,0]
                self.inflow_costs[t,:] += np.sum(midprops[:,np.newaxis]*((midas[:,np.newaxis] - t)*self.alpha + self.downnode.get_inflow_costs(self,midas)), axis=0)
    
    def reset(self):
        super().reset()
        
        self.flow_props = np.zeros_like(self.flows)
        self.flows_cum = np.zeros_like(self.flows_agg)
        
        self.A = np.stack([np.concatenate(([-1]*(cellid+1), np.arange(self.T-cellid).astype(int))) for cellid in np.arange(self.length)], axis=-1) # (T+1, length)
        self.a = np.stack([np.minimum(self.T, np.arange(self.T)+cellid+1) for cellid in np.arange(self.length)], axis=1)
        
class F3Source(Node):
    def __init__(self, inlinks, outlinks, nodeid, nodetype, demands, dtchoices):
        super().__init__(inlinks, outlinks, nodeid, nodetype)
        self.demands = demands
        self.dtchoices = dtchoices
        self.reset(dtchoices)
    
    def init_cost_computation(self):
        pass
    
    def determine_flows(self):
        pass
    
    def compute_costs(self):
        t = self.get_time()
        outlink = self.outlinks[0]
        self.outflow_costs[outlink][t,:] = outlink.inflow_costs[t,:]
    
    def reset(self, dtchoices):
        super().reset()
        self.dtchoices = dtchoices.copy()
        self.outflows[self.outlinks[0]] += self.demands*self.dtchoices
        
class F3Sink(Node):
    def __init__(self, inlinks, outlinks, nodeid, nodetype, sinkno):
        super().__init__(inlinks, outlinks, nodeid, nodetype)
        self.sinkno = sinkno
        
        self.alpha = pull_from_sim_info('alpha')
        self.beta = pull_from_sim_info('beta')
        self.gamma = pull_from_sim_info('gamma')
        self.tstar = pull_from_sim_info('tstar')
        self.M = pull_from_sim_info('M')
        
        self.reset()
    
    def init_cost_computation(self):
        inlink = self.inlinks[0]
        ts = np.arange(self.T)
        self.inflow_costs[inlink][:,:] = self.gamma*self.M
        self.inflow_costs[inlink][:,self.sinkno] = (self.beta[self.sinkno]*np.maximum(0, self.tstar[self.sinkno] - ts) + \
                                                    self.gamma[self.sinkno]*np.maximum(0, ts - self.tstar[self.sinkno]))
    
    def determine_flows(self):
        pass
    
    def compute_costs(self):
        pass
    
    def reset(self):
        super().reset()
        
class F3Merge(Node):
    def __init__(self, inlinks, outlinks, nodeid, nodetype):
        super().__init__(inlinks, outlinks, nodeid, nodetype)
        self.reset()
    
    def determine_flows(self):
        def calc_R_divided(inlinks_active, R_remaining):
            return (np.nan_to_num(inlinks_active*self.inlink_priorities/np.sum(inlinks_active*self.inlink_priorities))*R_remaining)
        
        t = self.get_time()
        outlink = self.outlinks[0]
        S = np.array([np.minimum(inlink.occs_agg[t,-1], inlink.Q) for inlink in self.inlinks])
        R = np.minimum(outlink.Q, outlink.w*(outlink.N - outlink.occs_agg[t,0]))
        
        flows_agg = np.zeros((self.in_degree))
        if float_le(np.sum(S), R):
            flows_agg += S
        else:
            inlinks_active = np.ones((self.in_degree,))
            R_remaining = R
            R_divided = calc_R_divided(inlinks_active, R_remaining)
            
            while np.any(float_lne(S, R_divided)):
                dem_constrained = float_lne(S, R_divided)
                flows_agg += dem_constrained*S
                R_remaining = R - np.sum(flows_agg)
                inlinks_active -= dem_constrained
                R_divided = calc_R_divided(inlinks_active, R_remaining)
            flows_agg += R_divided
        
        inflows = np.stack([inlink.disaggregate(-1,flows_agg[iterid]) for iterid,inlink in enumerate(self.inlinks)], axis=0)
        for iterid,inlink in enumerate(self.inlinks):
            self.inflows[inlink][t,:] = inflows[iterid,:]
        self.outflows[outlink][t,:] = np.sum(inflows, axis=0)
    
    def init_cost_computation(self):
        pass
    
    def compute_costs(self):
        t = self.get_time()
        outlink = self.outlinks[0]
        for inlink in self.inlinks:
            self.inflow_costs[inlink][t,:] = outlink.inflow_costs[t,:]
        self.outflow_costs[outlink][t,:] = outlink.inflow_costs[t,:]
    
    def reset(self):
        super().reset()
        tmp = np.array([inlink.Q for inlink in self.inlinks])
        self.inlink_priorities = tmp/np.sum(tmp)
        
class F3Diverge(Node):
    def __init__(self, inlinks, outlinks, nodeid, nodetype, srates):
        super().__init__(inlinks, outlinks, nodeid, nodetype)
        self.reset(srates)
    
    def determine_flows(self):
        t = self.get_time()
        inlink = self.inlinks[0]
        
        S = np.minimum(inlink.occs_agg[t,-1], inlink.Q)
        R = np.array([np.minimum(outlink.Q, outlink.w*(outlink.N - outlink.occs_agg[t,0])) for outlink in self.outlinks])
        
        h = inlink.A[t,-1]
        flows = np.zeros((self.out_degree, self.ntypes))
        if h < 0:
            return flows
        
        S_remaining = S
        R_remaining = R.copy()
        while (h < t+1-inlink.length):
            if h == inlink.A[t,-1]:
                iterflow_total = min(S, inlink.flows_cum[h,0] - inlink.flows_cum[t-1,-1])
            else:
                iterflow_total = min(S_remaining, inlink.flows_agg[h,0])
            
            iterflows = iterflow_total*inlink.flow_props[h,0,:]*self.srates[t,...]
            iterflows_link_agg = np.sum(iterflows, axis=-1)
            if np.all(float_le(iterflows_link_agg, R_remaining)):
                flows += iterflows
                S_remaining -= iterflow_total
                R_remaining -= np.minimum(iterflows_link_agg, R_remaining)
                
                if np.isclose(S_remaining, 0.) and ((not np.isclose(inlink.flows_cum[h,0], inlink.flows_cum[t-1,-1]+np.sum(flows))) or np.isclose(np.sum(flows), inlink.Q)):
                    break
                h += 1
            else:
                a = np.min(R_remaining/np.abs(iterflows_link_agg))
                flows += a*iterflows
                break
        inlink.A[t+1,-1] = h
        
        flows = np.round(flows, decimals=ROUND_DECIMALS)
        for iterid,outlink in enumerate(self.outlinks):
            self.outflows[outlink][t,:] = flows[iterid,:]
        self.inflows[inlink][t,:] = np.sum(flows, axis=0)
    
    def init_cost_computation(self):
        pass
    
    def compute_costs(self):
        t = self.get_time()
        inlink = self.inlinks[0]
        for iterid,outlink in enumerate(self.outlinks):
            self.inflow_costs[inlink][t,:] += self.srates[t,iterid,:]*outlink.inflow_costs[t,:]
            self.outflow_costs[outlink][t,:] = outlink.inflow_costs[t,:]
    
    def reset(self, srates):
        super().reset()
        self.srates = srates.copy()
        
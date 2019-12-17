import numpy as np
from scipy.optimize import minimize

EPS = 1e-08
MIN_PROP_VALUE = 1e-08

def opt_fn(x, choices, costs, lr):
    return 0.5*np.sum(np.power(x-(choices-lr*costs), 2))

def jac(x, choices, costs, lr):
    return (x-(choices-lr*costs))

def ones_jac(x):
    return np.ones_like(x)

def cons_fn(x):
    return np.sum(x)-1

def solveqp(choices, costs, lr):
    x = choices.copy()
    bounds = [(0,1)]*len(choices)
    constraint = {
        'type': 'eq',
        'fun': cons_fn,
        'jac': ones_jac
    }
    
    new_choices = minimize(opt_fn, x, args=(choices, costs, lr), jac=jac, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol':1e-09})
    return new_choices

def get_node_types(netgraph):
    sources = []
    sinks = []
    diverges = []
    merges = []
    
    for nodeid in netgraph.nodes:
        in_degree, out_degree = netgraph.in_degree(nodeid), netgraph.out_degree(nodeid)
        if (in_degree == 0) and (out_degree == 1):
            sources.append(nodeid)
        elif (in_degree == 1) and (out_degree == 0):
            sinks.append(nodeid)
        elif (in_degree == 1) and (out_degree > 1):
            diverges.append(nodeid)
        elif (in_degree > 1) and (out_degree == 1):
            merges.append(nodeid)
        else:
            print(in_degree, out_degree)
            raise 'Invalid node type!'
            
    return sources, sinks, diverges, merges

def float_le(a, b):
    return (a < b) | np.isclose(a, b, atol=EPS)

def float_ge(a, b):
    return (a > b) | np.isclose(a, b, atol=EPS)

def float_lne(a, b):
    return (a < b) & ~np.isclose(a, b, atol=EPS)

def float_gne(a, b):
    return (a > b) & ~np.isclose(a, b, atol=EPS)

def get_measures(new_dtchoices, new_srates, old_dtchoices, old_srates, dtccosts, srcosts):
    m1_list = []
    m3_list = []
    m5_list = []
    for k in old_dtchoices.keys():
        m1_list.append(np.sqrt(np.mean(np.square(new_dtchoices[k] - old_dtchoices[k]))))
        m3_list.append(np.mean((np.sum(old_dtchoices[k]*dtccosts[k], axis=0) - np.min(dtccosts[k], axis=0))/np.min(dtccosts[k], axis=0)))
        m5_list.append(np.mean(np.sum(old_dtchoices[k]*dtccosts[k], axis=0) - np.min(dtccosts[k], axis=0)))
    m1 = np.mean(m1_list)
    m3 = np.mean(m3_list)
    m5 = np.mean(m5_list)
    
    m2_list = []
    m4_list = []
    m6_list = []
    for k in old_srates:
        m2_list.append(np.sqrt(np.mean(np.square(new_srates[k] - old_srates[k]))))
        m4_list.append(np.mean((np.sum(old_srates[k]*srcosts[k], axis=1) - np.min(srcosts[k], axis=1))/np.min(srcosts[k], axis=1)))
        m6_list.append(np.mean((np.sum(old_srates[k]*srcosts[k], axis=1) - np.min(srcosts[k], axis=1))))
    m2 = np.mean(m2_list)
    m4 = np.mean(m4_list)
    m6 = np.mean(m6_list)
    
    return (m1, m2, m3, m4, m5, m6)



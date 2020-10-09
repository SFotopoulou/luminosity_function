#!/home/Sotiria/Software/anaconda3/bin/python
import sys
import time
sys.path.append('/home/Sotiria/Dropbox/AGNinCL/src/')


def run_MultiNest():
    import os, subprocess
    import json
    import matplotlib.pyplot as plt
    import pymultinest
    from AGN_LF_config import LF_config
    import Likelihood as lk
    
    ##########################################################################
    #Create output folder name
    folder_name = LF_config.outpath + "MultiNest" 
    if not os.path.exists(folder_name): os.mkdir(folder_name)
    print('results in: ', folder_name)
    
    
    def show(filepath):
        """ open the output (pdf) file for the user """
        if os.name == 'mac': subprocess.call(('open', filepath))
        elif os.name == 'nt': os.startfile(filepath)
        elif os.name == 'posix': subprocess.call(('xdg-open', filepath))
    
    if LF_config.model == 'LDDE':
        parameters = [ "L0", "g1", "g2", "p1", "p2", "zc", "La", "a", "Norm"]
        Phi = lk.Fotopoulou_Likelihood
        
    n_params = len(parameters)
    
    ##########################################################################
    # Prior definition
    def Uniform(r,x1,x2):
        return x1+r*(x2-x1)
    
    def myUniformPrior(cube, ndim, nparams):
            
        if LF_config.model == 'LDDE':
            cube[0] = Uniform(cube[0], 40.0, 47.0) # L0        
            cube[1] = Uniform(cube[1], -2.0, 1.5) # g1
            cube[2] = Uniform(cube[2], 1.5, 5.0) # g2
            cube[3] = Uniform(cube[3], 0.0, 10.0) # p1        
            cube[4] = Uniform(cube[4], -20.0, 5.0) # p2        
            cube[5] = Uniform(cube[5], 0.0, 6.0)  # zc
            cube[6] = Uniform(cube[6], 40.0, 47.0) # La
            cube[7] = Uniform(cube[7], 0.0, 1.0) # a
            cube[8] = Uniform(cube[8], -10.0, -2.0) # Norm
    
    # Likelihood    
    def myLogLike(cube, ndim, nparams):
        params = [cube[i] for i in range(0, nparams)]
        LogL = Phi(*params)
        return -0.5*LogL
    
    ##########################################################################
    print("\nRunning MultiNest\n")
    t1 = time.time()
    pymultinest.run(myLogLike,
                    myUniformPrior,
                    n_dims = n_params,
                    n_params = n_params,
                    importance_nested_sampling=False,
                    multimodal=True, const_efficiency_mode=False, n_live_points=1000,
                    evidence_tolerance=0.5,
                    n_iter_before_update=100, null_log_evidence=-1e90, 
                    max_modes=100, mode_tolerance=-1e90, 
                    seed=-1, 
                    context=0, write_output=True, log_zero=-1e100, 
                    max_iter=0, init_MPI=False,
                    outputfiles_basename = folder_name+"/1-",
                    resume = False,
                    verbose = True,
                    sampling_efficiency = 'parameter')
    print("Multinest done in ", time.time()-t1,"sec")
        
    # lets analyse the results
    a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = folder_name+"/1-")
    s = a.get_stats()
    
    json.dump(s, open('%s.json' % a.outputfiles_basename, 'w'), indent=2)
    print()
    print("-" * 30, 'ANALYSIS', "-" * 30)
    print("Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] ))
    plt.clf()
    
    p = pymultinest.PlotMarginalModes(a)
    plt.figure(figsize=(5*n_params, 5*n_params))
    for i in range(n_params):
        plt.subplot(n_params, n_params, n_params * i + i + 1)
        p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
        plt.ylabel("Probability")
        plt.xlabel(parameters[i])
    
        for j in range(i):
            plt.subplot(n_params, n_params, n_params * j + i + 1)
            p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
            plt.xlabel(parameters[i])
            plt.ylabel(parameters[j])
    
    plt.savefig(folder_name+"/marginals_multinest.pdf")
    
    for i in range(n_params):
        outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
        p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
        plt.ylabel("Probability")
        plt.xlabel(parameters[i])
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()
    
        outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
        p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
        plt.ylabel("Cumulative probability")
        plt.xlabel(parameters[i])
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()
    
    print( "take a look at the pdf files in chains/" )
    
    
if __name__ == "__main__":
 
    import time
    t1 = time.time()
    run_MultiNest()
    print(time.time()-t1,'s')
   
    
    
    

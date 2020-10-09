#!/home/Sotiria/Software/anaconda3/bin/python

if __name__ == "__main__":
    # default values
    from AGN_LF_config import LF_config

    # overwrite default values such as fields, paths,etc using argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Define parameters for XLF batch run.')
    
    
    
    # call preferred method
    from MLE import run_MLE
    from MultiNest import run_MultiNest
    from Vmax import Vmax
    
    
    

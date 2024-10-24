import os

class Parameters(object):
    # Rooths for Models in Jupyter
    root_path              = os.path.abspath(os.path.join('../'+os.path.dirname('__file__')))
    input_path             = os.path.join(root_path, 'data', 'input')
    output_path            = os.path.join(root_path, 'data', 'output')
    raw_path               = os.path.join(root_path, 'data', 'input','raw')
    results_path           = os.path.join(root_path, 'data', 'output', 'results')
    models_path            = os.path.join(root_path, 'data', 'output', 'models')
    plots_path             = os.path.join(root_path, 'data', 'output', 'plots')
    
    # Rooths for API
    root_path_api          = os.path.abspath(os.path.join(os.path.dirname('__file__')))
    results_path_api       = os.path.join(root_path_api, 'data', 'output', 'results')
    
    # General
    steps                  = 4
    col_pronos             = 'Demand'
    col_y                  = 'year_month'
    scales                 = [
                            col_pronos+'_og',
                            col_pronos+'_scal01',
                            col_pronos+'_scal11',
                            col_pronos+'_scallg'
                            ]
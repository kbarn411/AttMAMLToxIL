from maml_loop_codes import *

config_dict = {
    "descriptors_type" : 'twod', # 'twod' for RDKIT or 'tr' for CHEMBERTA
    
    "transform_y" : True,
    "flag_per_task_transform" : True,
    "apply_log_flag" : False,
    "apply_power_flag" : True,
    "power_transform_type" : 'robust',
    "apply_minmax_flag" : True,

    "flag_verbose" : 0,

    "flag_compare_to_onetask_ml" : True,
    "flag_use_whole_ds_for_classical" : False, 
    
    "flag_use_ifm" : True,
    "flag_use_similar_tasks" : False, 
    "flag_test_on_new_tasks" : False,
    "flag_use_same_units" : False,
    "flag_two_losses" : False,

    "flag_attmaml" : True,
    
    "flag_subseta_only" : True, 
    
    "flag_use_validation" : 1,
    "root_dir" : "./",

    "base_n_of_episodes" : 1200,
    "seed_values" : [123],
    "validation_schema_to_test" : 'random', 
    "shot_sizes_to_test" : [32],

    "meta_batch_size" : 2,
    "optimization_steps" : 1, 

    "list_with_ft_episodes" : [*list(range(1,201,10)), *list(range(201,2501,100))],
}


execute_experiment(config_dict)

import yaml
for feature_fraction in [0.5,0.6,0.7]:
    for bagging_fraction in [0.5,0.6,0.7]:
        for bagging_freq in [3, 4, 5]:
            for depth in [2,3,4]:
            	d = {"feature_fraction":feature_fraction, "bagging_fraction": bagging_fraction, \
                "bagging_freq": bagging_freq, "max_depth": depth}
            	with open('./config_3/{}_{}_{}_{}.yaml'.format(str(depth), str(feature_fraction), str(bagging_fraction),str(bagging_freq)), 'w') as yaml_file:
                	yaml.dump(d, yaml_file, default_flow_style=False)
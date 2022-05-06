import yaml
for lr in [0.01,0.025,0.05]:
    for feature_fraction in [0.8,0.9,1.0]:
        for bagging_fraction in [0.8,0.9,1.0]:
            for bagging_freq in [3, 4, 5]:
                d = {"learning_rate":lr, "feature_fraction":feature_fraction, "bagging_fraction": bagging_fraction, "bagging_freq": bagging_freq}
                with open('./config_1/{}_{}_{}_{}.yaml'.format(str(lr), str(feature_fraction), str(bagging_fraction),str(bagging_freq)), 'w') as yaml_file:
                    yaml.dump(d, yaml_file, default_flow_style=False)
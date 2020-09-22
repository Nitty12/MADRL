

To create a new environment to run this project: 

    conda create --name test_env --file conda_requirements.txt

activate the new environment and do

    pip install -r pip_requirements.txt

(If there is error for tensorflow, install tensorflow-estimator version 2.1.0)

Install the Local flexibility market gym environment

goto folder gym-LocalFlexMarketEnv and 

    pip install -e .

copy the inputs folder

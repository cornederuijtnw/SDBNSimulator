SDBNSimulator - Simulation model for SDBN
====

Simulation model for SDBN

# How to Use

## Installation
To install the SDBNSimulator, run the following code in command line:

```bash
cd $PROJECT_DIR
sudo python setup.py install
```

Dependencies:

* pandas
* deprecated
* numpy
* scikit-learn

## Running examples
The underlying code runs the simulation for baseline parameters. The parameters can be adjusted by adjusting the
parameters in ./allcode/create_simulation_settings.py.

```bash
chmod +x ./run_analysis.sh
./run_analysis.sh $TEST_FRAC $SEED
```


#INPUT PARAMETERS:
# RUN_FLEDER: should the Fleder simulation be run?

# Activate the virtual environment
# conda activate SDBNSimulator
# Example call: ./allcode/managers/run_analysis.sh FALSE ./data/simulation_inputs/base_simulation_case.pl

TEST_FRAC=$1
SPLIT_SEED=$2

echo "Creating simulation settings"
python -W ignore ./create_simulation_settings.py

# Run the simulation:
echo 'Running the fleder simulation'
python -W ignore ./fleder_simulation.py $TEST_FRAC $SPLIT_SEED
echo "Fleder simulation completed"



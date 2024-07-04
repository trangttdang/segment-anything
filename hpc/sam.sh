#!/bin/bash -l

#SBATCH --job-name=drone
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=2024-spring-cs-370-monogiou-ttd22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:59:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M

module load foss/2021b Python/3.9.6


# ACTIVATE VENV

echo "Activating virtual environment"
source venv/bin/activate

# INSTALL REQUIREMENTS

PIP3=false
if [ -x "$(command -v pip3)" ]; then
    PIP3=true
fi

echo "Installing requirements"
if [ $PIP3 == true ]; then
    echo "use pip3 to download"
    pip3 install --upgrade pip
    # pip3 install json
    pip3 install datasets
    pip3 install torch
    pip3 install transformers
    pip3 install tqdm
    pip3 install statistics
    pip3 install monai
    pip3 install pillow
    pip3 install numpy
    pip3 install matplotlib
    pip3 install -q -r requirements.txt
else
    echo "use pip to download"
    pip install -q -r requirements.txt
fi

# RUN THE MODEL

echo "Running the script"
if [ $PYTHON3 == true ]; then
    python3 sam.py
else
    python sam.py
fi
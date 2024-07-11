## This is an example of an sbatch script to run a tensorflow script
## using Singularity to run the tensorflow image.
##
## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Change reserved resources as needed for your job.
##
#SBATCH --job-name=tensorflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=run-%j.log
unset OMP_NUM_THREADS
# Set SIMG path
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.3.1/simg
# Set SIMG name
SIMG_NAME=tensorflow2.3.1-py3-cuda10.1-ubuntu18.04.simg
# Set data path
DATA_PATH=/pine/scr/m/w/mwaldron
# GPU with Singularity
singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "cd $DATA_PATH; ./test.py"
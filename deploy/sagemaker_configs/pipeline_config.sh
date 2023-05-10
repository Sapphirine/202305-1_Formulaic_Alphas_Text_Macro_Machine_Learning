set -e
ENVIRONMENT=python3

#Declare all the jupyter notebooks that need to run, within the Sagemaker instance
FILE0="/home/ec2-user/SageMaker/0. Fetch Latest Market Data.ipynb"
FILE1="/home/ec2-user/SageMaker/1. Machine Learning Purged Cross Val Grid Search.ipynb"
FILE2="/home/ec2-user/SageMaker/2. Best Param Out of Fold.ipynb"
FILE3="/home/ec2-user/SageMaker/3. Interpretable Machine Learning.ipynb"

#Activate python environment. The lifecycle configuration cannot autodetect the environment
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

#Install packages here instead of putting them inside the notebooks
pip install --upgrade pip
pip install yfinance

#Execute the notebook in background
nohup jupyter nbconvert "$FILE0" "$FILE1" "$FILE2" "$FILE3" --ExecutePreprocessor.kernel_name=python3 --to notebook --inplace --execute

source /home/ec2-user/anaconda3/bin/deactivate

# PARAMETERS
IDLE_TIME=60
AUTO_STOP_FILE="/home/ec2-user/SageMaker/auto-stop.py"

echo "Fetching the autostop script"
wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py

echo "Starting the SageMaker autostop script in cron"
(crontab -l 2>/dev/null; echo "*/1 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab -
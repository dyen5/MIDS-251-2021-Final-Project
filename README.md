# MIDS-251-2021-Final-Project

## S3 Bucket
```
s3://w251-covidx-ct
```

## AWS EC2 Instance
```
ami-042e8287309f5df03
tx2.large
Ubuntu 20.04
Volume = 128 GB

SSH port 22 anywhere
HTTPS port 443 anywhere
custom TCP port 8888 anywhere
```
## Dependencies
```
sudo su
apt update
apt install python3-pip
apt install awscli
apt install unzip

pip3 install kaggle
```

# Get Data from Kaggle

Follow directions from the following site to get the dataset from Kaggle
```
https://confusedcoders.com/data-engineering/how-to-copy-kaggle-data-to-amazon-s3
```

An appreviated version of the instructions is as follows:

Create Kaggle directory
```
mkdir ~/.kaggle
```

Get Kaggle token from Kaggle Account.  Create kaggle.json and copy token into file.
```
cd .kaggle
vim kaggle.json
```

Download Dataset using Kaggle API - 10 minutes 
```
kaggle datasets download -d hgunraj/covidxct
```

Unzip dataset - 10 minutes
```
unzip covidxct.zip
```

Configure AWS
```
aws configure 
Access Key ID:  ####################
Secret Acess Key: ##################
Default region name: us-east-1
Default output format: <hit enter>
```

Copy files to S3 Bucket
```
aws s3 cp metadata.csv s3://w251-covidx-ct
aws s3 cp train_COVIDx_CT-2A.txt s3://w251-covidx-ct
aws s3 cp val_COVIDx_CT-2A.txt s3://w251-covidx-ct
aws s3 cp test_COVIDx_CT-2A.txt s3://w251-covidx-ct
aws s3 cp covidxct.zip s3://w251-covidx-ct
```

This copies the each image in the local file and places in a folder called 2A_images on s3 - 20 minutes
```
aws s3 cp 2A_images s3://w251-covidx-ct/2A_images --recursive
```

Check contents in s3
```
aws s3 ls s3://w251-covidx-ct
```

Sync images from s3 bucket to local machine
```
aws s3 sync s3://w251-covidx-ct /tmp/2A_images
```

# Setup Jupyter Notebook
```
https://dataschool.com/data-modeling-101/running-jupyter-notebook-on-an-ec2-server/
```

Download Anaconda to the VM
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

Enter yes for legal information and yes for appending path
```
export PATH=/root/anaconda3/bin:$PATH
```

If no was accidentally pressed, edit the path by (scroll all the way down). note: the bashrc file is hidden.
```
cd ~/root/andaconda3
vi ~/.bashrc
source ~/.bashrc
```

Exit terminal and ssh back into instance to update paths

Enter ipython in commandline
```
ipython
```

Set a password for the notebook (password=w251
```
from IPython.lib import passwd

passwd()
exit
```

Copy the hash code from the output

Edit the jupyter notebook config file
```
cd ~/.jupyter

vim jupyter_notebook_config.py_
```

Copy the following to the top of the config file and edit the password hash
```
conf = get_config()

conf.NotebookApp.ip = '0.0.0.0'
conf.NotebookApp.password = u'<YOUR PASSWORD HASH>'
conf.NotebookApp.port = 8888
```

Cd to the directory of the datafile and start the jupyter notebook
```
cd ~/.kaggle

jupyter notebook
http://ec2-54-89-73-47.compute-1.amazonaws.com:8888/
```

# References

### Create credentials in AWS
https://aws.amazon.com/premiumsupport/knowledge-center/s3-locate-credentials-error/

### s3 bucket commands
https://www.thegeekstuff.com/2019/04/aws-s3-cli-examples/

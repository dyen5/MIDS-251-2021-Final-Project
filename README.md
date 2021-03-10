# MIDS-251-2021-Final-Project

## AWS EC2 Instance
```
ami-042e8287309f5df03
tx2.large
Ubuntu 20.04
Volume = 128 GB
```
## Dependencies
```
apt update
apt install python3-pip
apt install aws cli
apt unstall unzip

pip3 install kaggle
```
## Configure AWS
```
aws configure 
```

# Get Data from Kaggle

Follow directions from the following site to get the dataset from Kaggle
```
https://confusedcoders.com/data-engineering/how-to-copy-kaggle-data-to-amazon-s3
```

An appreviated version of the instructions is as follows:

Create Kaggle directory
```
mkdir .kaggle
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

# Setup Jupyter Notebook
```
https://dataschool.com/data-modeling-101/running-jupyter-notebook-on-an-ec2-server/
```

# References

### Create credentials in AWS
https://aws.amazon.com/premiumsupport/knowledge-center/s3-locate-credentials-error/

### Uploading data from Kaggle to S3 Bucket
S3 bucket name = w251-covidx-ct

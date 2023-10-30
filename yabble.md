## Yabble Deployment notes

Need to:

1. Find the right EC2 instance with the right virtual machine image
2. Create a Dockerfile that can be deployed on this instance

## Manually via Web Interface

1. [EC2 console](https://ap-southeast-2.console.aws.amazon.com/ec2/home?region=ap-southeast-2#Home)
2. Choose "Launch Instance"
3. Choose an AMI, search for "Deep Learning". Choose something like "Deep Learning AMI GPU PyTorch 2.0.1"
4. Choose a machine type "g4dn.2xlarge"
5. Connect via SSH

# Via command line

## Find the right instance type (with GPU)

find a machine AMI that matches our specs i.e. CUDA 11.8
from : https://aws.amazon.com/releasenotes/aws-deep-learning-ami-catalog/
run:


List of available machine types in ap-southeast-2 zone run:

	aws ec2 describe-instance-type-offerings --location-type "availability-zone" --filters Name=location,Values=ap-southeast-2b --region ap-southeast-2 --query "InstanceTypeOfferings[*].[InstanceType]" --output text | sort

Returns a big list. These as the GPU instances types:


    g3.16xlarge
    g3.4xlarge
    g3.8xlarge
    g3s.xlarge
    g4dn.12xlarge
    g4dn.16xlarge
    g4dn.2xlarge
    g4dn.4xlarge
    g4dn.8xlarge
    g4dn.metal
    g4dn.xlarge
    inf1.24xlarge
    inf1.2xlarge
    inf1.6xlarge
    inf1.xlarge

look under Accelerated Computing here: https://aws.amazon.com/ec2/instance-types/
Also need to check [the prices](https://aws.amazon.com/ec2/pricing/on-demand/?refid=a5f6fe36-4e3c-4c1b-a71a-b795de7b4ed7).

## Find the right virtual machine (AMI) to use with our docker container.

Machine needs to have the same CUDA driver (minimum) version installed as the container needs in this case 11.8

From: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-catalog/

Run this command to find the AMI ID for our zone

	aws ec2 describe-images --region us-east-1 --owners amazon --filters 'Name=name,Values=AWS Deep Learning Base AMI GPU CUDA 11 (Ubuntu 20.04) ????????' 'Name=state,Values=available' --query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text

Which outputs

	ami-0182289040a1ed516

Start to create an instance by hand from the EC2 UI

https://ap-southeast-2.console.aws.amazon.com/ec2/home?region=ap-southeast-2#LaunchInstances:


## Dockerfile setup

copided Dockerfile from here https://github.com/Wordcab/wordcab-transcribe/blob/main/Dockerfile

Added the setup for pyTorch manually to match the CUDA version.


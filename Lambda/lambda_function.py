import boto3
import email
import json
import urllib.parse
from datetime import datetime
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

region = 'us-east-1'

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('runtime.sagemaker')
ses_client = boto3.client('ses', region_name=region)

def lambda_handler(event, context):

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    print(event['Records'][0]['s3']['bucket']['name'])
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response['Body'].read().decode('utf-8')
        
        # Extract contents from email
        email_contents = email.message_from_string(body)
        
        email_datetime = email_contents.get('Date')
        email_datetime = email_datetime[:email_datetime.find('-')-1]
        dt = datetime.strptime(email_datetime, '%a, %d %b %Y %H:%M:%S')
        email_date = str(dt.date())
        email_time = str(dt.time())
        
        email_recipient = email_contents.get('To')
        email_sender = email_contents.get('From')
        email_sender = email_sender[email_sender.find('<')+1:-1]
        email_subject = email_contents.get('Subject')
        
        email_body = ''
        if email_contents.is_multipart():
            for payload in email_contents.get_payload():
                if payload.get_content_type() == 'text/plain':
                    email_body = payload.get_payload()
        else:
            email_body = email_contents.get_payload()
            
        email_body = email_body.replace("\r", " ").replace("\n", " ")
        
        # Prepare input for sagemaker endpoint
        endpoint_name = 'sms-spam-classifier-mxnet-2022-11-17-23-51-03-470'
        detector_input = [email_body]
        
        vocabulary_length = 9013
        one_hot_detector_input = one_hot_encode(detector_input, vocabulary_length)
        encoded_detector_input = vectorize_sequences(one_hot_detector_input, vocabulary_length)
        detector_input = json.dumps(encoded_detector_input.tolist())
        
        # Get a response from the sagemaker endpoint and decode it
        response = sagemaker_client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=detector_input)
        
        results = response['Body'].read().decode('utf-8')
        results_json = json.loads("" + results + "")
        
        # Get the class and confidence percentage
        if(results_json['predicted_label'][0][0]==1.0):
            spam_class = 'SPAM'
        else:
             spam_class = 'HAM'
        confidence_score = str(results_json['predicted_probability'][0][0]*100)
        confidence_score = confidence_score.split('.')[0]
        
        # Send the email through SES
        SES_email_body = email_body
        if len(SES_email_body) > 240:
            SES_email_body = SES_email_body[:240]
        
        SES_email_line1 = 'We received your email sent on ' + email_date + ' at ' + email_time + ' with the subject ' + email_subject + '.\n\n'
        SES_email_line2 = 'Here is a 240 character sample of the email body:\n'
        SES_email_line3 = SES_email_body + '\n\n'
        SES_email_line4 = 'The email was categorized as ' + spam_class + ' with a ' + confidence_score + '% confidence.'
        
        SES_email = SES_email_line1 + SES_email_line2 + SES_email_line3 + SES_email_line4
        
        charset = "UTF-8"
        response = ses_client.send_email(
            Destination={
                "ToAddresses": [
                    email_sender,
                ],
            },
            Message={
                "Body": {
                    "Text": {
                        "Charset": charset,
                        "Data": SES_email,
                    }
                },
                "Subject": {
                    "Charset": charset,
                    "Data": "Spam Detector Results",
                },
            },
            Source="regan@reganjbragg.tech",
        )
        
        return "success"
    
    except Exception as e:
        print(e)
        raise e

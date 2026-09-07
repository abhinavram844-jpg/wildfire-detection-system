import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv() # Load environment variables from a .env file

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

client = Client(account_sid, auth_token)
message = client.messages.create(
    body ="I love Python",
    from_ = "+17372212163",
    to = "+919840082807"   
)

print(message.sid)
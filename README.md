# Make sure to run these before starting anything at the first run
- python manage.py makemigrations
- python manage.py migrate
- python manage.py createsuperuser ( create a admin account )

# Django Cheque clearance server
This is a cheque clearance server designed in django to detect, extract details and simulate clearance of cheques which are scanned using phone cameras.

## To run (may need to add executable permission for server file):
- ./server sbi ( or nic)
- 
- Go to localhost:{port}/admin
- Login to the created account
- create a user and a bank account


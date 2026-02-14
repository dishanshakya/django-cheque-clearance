# Make sure to run these before starting anything at the first run
- python manage.py makemigrations
- python manage.py migrate
- python manage.py createsuperuser

# Django Cheque clearance server
This is a cheque clearance server designed in django to detect, extract details and simulate clearance of cheques which are scanned using phone cameras.

## To run (may need to add executable permission for server file):
- ./server sbi ( or nic)
- 
- Go to localhost:{port}/admin
- create a user and a bank account


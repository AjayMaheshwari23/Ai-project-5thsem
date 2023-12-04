import smtplib
from email.message import EmailMessage


def email_alert(to):
    msg = EmailMessage()
    msg.set_content('I just fell down! I might be unconscious. \nVisit me at my house and please call an ambulance if you are free.')
    msg['subject'] = 'Emergency Alert by Mr. X!'
    msg['to'] = to

    user = ''
    msg['from'] = user
    password = ''

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()

email_alert(receiver)

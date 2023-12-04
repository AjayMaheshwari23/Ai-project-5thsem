import smtplib
from email.message import EmailMessage


def email_alert(to):
    msg = EmailMessage()
    msg.set_content('I fell! Nothing serious, just might be unconsious... \nVisit me at my room if possible and call an ambulance if you are free. \n\nP.S. I might be dead till you get here...'
)
    msg['subject'] = 'Howdy mate!'
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